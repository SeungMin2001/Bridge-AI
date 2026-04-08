"""
MergePRAG 추론 모듈
- 과목별 메모리 관리 (K,V 캐시)
- Orthogonal Merging으로 passage 병합
- Critical Layer Hook Inject
"""
import torch
import io
from .cross_attention import cross_attention
from .hypernetwork import HyperNetwork
from .orthogonal_merge import orthogonal_merging

# ── 설정 (train.py와 동일) ──
CRITICAL_LAYER = 0
NUM_KV = 16
import os as _os
WEIGHTS_PATH = _os.path.join(_os.path.dirname(__file__), "hypernet_weights.pt")


def make_hook(delta_K, delta_V):
    """Critical Layer에 K,V를 inject하는 forward hook (train.py 방식)"""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            return (hidden + delta,) + output[1:]
        else:
            hidden = output
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            return hidden + delta
    return hook_fn


class CourseMemoryManager:
    """과목별 MergePRAG 메모리 관리"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        d_model = model.config.hidden_size

        # HyperNetwork 로드
        self.hypernet = HyperNetwork(d_model, k=NUM_KV).to(device).float()
        self.hypernet.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device)
        )
        self.hypernet.eval()
        print(f"[MergePRAG] HyperNetwork 로드 완료 (d_model={d_model}, k={NUM_KV})")

        # 과목별 메모리 캐시: {course_id: {"K": Tensor, "V": Tensor, "count": int}}
        self.memories = {}

    def encode_passage(self, passage: str):
        """passage → Qwen embed_tokens → HyperNetwork → K, V"""
        with torch.no_grad():
            input_ids = self.tokenizer(
                passage, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"].to(self.device)
            c_emb = self.model.model.embed_tokens(input_ids)  # [1, T, d_model]
            c_emb = c_emb.to(dtype=torch.float32)
            K, V = self.hypernet(c_emb)  # [1, NUM_KV, d_model]
        return K, V

    def add_passage(self, course_id: str, passage: str):
        """passage를 과목 메모리에 orthogonal merge로 추가"""
        new_K, new_V = self.encode_passage(passage)

        if course_id not in self.memories:
            # 첫 passage: 그대로 저장
            self.memories[course_id] = {
                "K": new_K.squeeze(0),  # [NUM_KV, d_model]
                "V": new_V.squeeze(0),
                "count": 1,
            }
        else:
            # orthogonal merge로 기존 메모리에 병합
            mem = self.memories[course_id]
            mem["K"] = orthogonal_merging(mem["K"], new_K.squeeze(0))
            mem["V"] = orthogonal_merging(mem["V"], new_V.squeeze(0))
            mem["count"] += 1

        return self.memories[course_id]["count"]

    def add_passages(self, course_id: str, passages: list[str]):
        """여러 passage를 한번에 과목 메모리에 추가"""
        for p in passages:
            self.add_passage(course_id, p)
        return self.memories.get(course_id, {}).get("count", 0)

    def get_memory(self, course_id: str):
        """과목의 merged K, V 반환. 없으면 None"""
        mem = self.memories.get(course_id)
        if mem is None:
            return None, None
        return mem["K"].unsqueeze(0), mem["V"].unsqueeze(0)  # [1, NUM_KV, d_model]

    def has_memory(self, course_id: str) -> bool:
        return course_id in self.memories

    def clear_memory(self, course_id: str):
        """과목 메모리 초기화"""
        if course_id in self.memories:
            del self.memories[course_id]

    def list_courses(self) -> list[dict]:
        """모든 과목 메모리 상태 조회"""
        return [
            {"course_id": cid, "passage_count": mem["count"]}
            for cid, mem in self.memories.items()
        ]

    # ── DB 직렬화/역직렬화 ──
    @staticmethod
    def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf)
        return buf.getvalue()

    @staticmethod
    def bytes_to_tensor(data: bytes, device) -> torch.Tensor:
        buf = io.BytesIO(data)
        return torch.load(buf, map_location=device)

    def save_to_db(self, course_id: str, conn):
        """과목 메모리를 DB에 저장"""
        mem = self.memories.get(course_id)
        if mem is None:
            return
        import uuid
        from datetime import datetime
        k_bytes = self.tensor_to_bytes(mem["K"])
        v_bytes = self.tensor_to_bytes(mem["V"])

        conn.execute("""
            INSERT INTO course_memories (memory_id, course_id, merged_k, merged_v, passage_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (course_id) DO UPDATE SET
                merged_k = EXCLUDED.merged_k,
                merged_v = EXCLUDED.merged_v,
                passage_count = EXCLUDED.passage_count,
                updated_at = EXCLUDED.updated_at
        """, (str(uuid.uuid4()), course_id, k_bytes, v_bytes, mem["count"], datetime.now()))
        conn.commit()

    def load_from_db(self, course_id: str, conn):
        """DB에서 과목 메모리 로드"""
        cur = conn.cursor()
        cur.execute(
            "SELECT merged_k, merged_v, passage_count FROM course_memories WHERE course_id = %s",
            (course_id,)
        )
        row = cur.fetchone()
        cur.close()
        if row and row[0] and row[1]:
            self.memories[course_id] = {
                "K": self.bytes_to_tensor(row[0], self.device),
                "V": self.bytes_to_tensor(row[1], self.device),
                "count": row[2],
            }
            return True
        return False
