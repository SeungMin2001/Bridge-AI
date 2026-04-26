"""
MergePRAG 추론 모듈
- 과목별 메모리 관리 (K,V 캐시)
- Orthogonal Merging으로 passage 병합
- Critical Layer Hook Inject
"""
import torch
import io
import os
from .cross_attention import cross_attention
from .config import (
    ALPHA,
    NUM_KV,
    USE_CONTEXTUAL_PASSAGE_ENCODER,
    USE_QUESTION_CONDITIONED_MEMORY,
    load_critical_layer,
    load_hypernet_state_dict,
)
from .embedding import encode_passage_states, tokenize_conditioned_memory
from .hypernetwork import HyperNetwork
from .orthogonal_merge import orthogonal_merging

# ── 설정 (train.py와 동일) ──
CRITICAL_LAYER = load_critical_layer()


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def make_hook(delta_K, delta_V, alpha=ALPHA):
    """Critical Layer에 K,V를 inject하는 forward hook"""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            return (hidden + alpha * delta,) + output[1:]
        else:
            hidden = output
            K = delta_K.to(device=hidden.device, dtype=hidden.dtype)
            V = delta_V.to(device=hidden.device, dtype=hidden.dtype)
            delta = cross_attention(hidden, K, V)
            return hidden + alpha * delta
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
        state_dict, load_info = load_hypernet_state_dict(map_location=device)
        self.hypernet.load_state_dict(state_dict)
        self.hypernet.eval()
        source = load_info["source"]
        step = load_info["step"]
        kind = load_info["kind"]
        step_text = f", step={step}" if step is not None else ""
        print(
            f"[MergePRAG] HyperNetwork 로드 완료 "
            f"(d_model={d_model}, k={NUM_KV}, question_conditioned={USE_QUESTION_CONDITIONED_MEMORY}, "
            f"source={source}, kind={kind}{step_text})"
        )

        # 과목별 메모리 캐시: {course_id: {"K": Tensor, "V": Tensor, "count": int}}
        self.memories = {}

    def encode_passage(self, passage: str, question: str | None = None):
        """Build K,V from a passage, optionally conditioned on the current question."""
        with torch.no_grad():
            should_condition = bool(question) and USE_QUESTION_CONDITIONED_MEMORY
            if should_condition:
                encoded = tokenize_conditioned_memory(
                    self.tokenizer,
                    question,
                    passage,
                    self.device,
                    max_length=512,
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                question_mask = encoded["question_mask"]
                passage_mask = encoded["passage_mask"]
                query_focus_mask = encoded.get("query_focus_mask")
            else:
                encoded = self.tokenizer(
                    passage, return_tensors="pt", truncation=True, max_length=512
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                question_mask = None
                passage_mask = None
                query_focus_mask = None

            c_emb = encode_passage_states(
                self.model,
                input_ids,
                attention_mask=attention_mask,
                use_contextual=USE_CONTEXTUAL_PASSAGE_ENCODER,
            )
            query = masked_mean(c_emb, question_mask) if question_mask is not None else None
            K, V = self.hypernet(
                c_emb,
                attention_mask=attention_mask,
                query=query,
                focus_mask=passage_mask,
                query_focus_mask=query_focus_mask,
            )
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
                "passages": [passage],
            }
        else:
            # orthogonal merge로 기존 메모리에 병합
            mem = self.memories[course_id]
            mem["K"] = orthogonal_merging(mem["K"], new_K.squeeze(0))
            mem["V"] = orthogonal_merging(mem["V"], new_V.squeeze(0))
            mem["count"] += 1
            mem.setdefault("passages", []).append(passage)

        return self.memories[course_id]["count"]

    def add_passages(self, course_id: str, passages: list[str]):
        """여러 passage를 한번에 과목 메모리에 추가"""
        for p in passages:
            self.add_passage(course_id, p)
        return self.memories.get(course_id, {}).get("count", 0)

    def get_memory(self, course_id: str, question: str | None = None):
        """과목의 merged K, V 반환. 없으면 None"""
        mem = self.memories.get(course_id)
        if mem is None:
            return None, None
        passages = mem.get("passages") or []
        if question and passages and USE_QUESTION_CONDITIONED_MEMORY:
            merged_k = None
            merged_v = None
            for passage in passages:
                cur_k, cur_v = self.encode_passage(passage, question=question)
                cur_k = cur_k.squeeze(0)
                cur_v = cur_v.squeeze(0)
                if merged_k is None:
                    merged_k = cur_k
                    merged_v = cur_v
                else:
                    merged_k = orthogonal_merging(merged_k, cur_k)
                    merged_v = orthogonal_merging(merged_v, cur_v)
            return merged_k.unsqueeze(0), merged_v.unsqueeze(0)
        if question and USE_QUESTION_CONDITIONED_MEMORY and not passages:
            # Question-conditioned memory must be regenerated from original passages.
            # Stored K/V alone is not enough because a different question changes K/V.
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
                "passages": [],
            }
            return True
        return False
