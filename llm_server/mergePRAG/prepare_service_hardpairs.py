"""
Create service-oriented hard-pair data for MergePRAG.

The generated rows target the real product behavior:
given a lecture utterance passage, the injected memory must make the model
answer from that passage, even when a near-identical passage flips the answer.

Output schema matches MergePRAGDataset:
  {
    "source_id": "...",
    "task": "final_qa",
    "question": "...",
    "answer": "...",
    "passage": "...",
    "contrast_id": "...",
    "hard_negatives": [{"passage": "...", "answer": "..."}]
  }

Usage:
  python -m llm_server.mergePRAG.prepare_service_hardpairs
  python -m llm_server.mergePRAG.prepare_service_hardpairs --output-dir C:\\Users\\user\\Documents\\last_project\\data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .eval_cases import SERVICE_DIAGNOSTIC_CASE


PROJECT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
WINDOWS_DATA_DIR = Path(r"C:\Users\user\Documents\last_project\data")
DEFAULT_OUTPUT_DIR = PROJECT_DATA_DIR if PROJECT_DATA_DIR.is_dir() else WINDOWS_DATA_DIR
HANGUL_RE = re.compile(r"[가-힣]")


EN_TEAMS = [
    ("Manchester United", "Chelsea"),
    ("Bristol Rovers", "Bristol Bears"),
    ("Seoul Tigers", "Busan Waves"),
    ("New York Red Bulls", "Chicago Fire"),
    ("Daejeon Falcons", "Incheon Mariners"),
]

KO_TEAMS = [
    ("맨체스터 유나이티드", "첼시"),
    ("서울 타이거즈", "부산 웨이브스"),
    ("대전 팔콘스", "인천 매리너스"),
    ("컴퓨터공학과 A팀", "데이터사이언스 B팀"),
    ("1분반", "2분반"),
]

DEADLINES_EN = [
    ("homework", "Monday", "Friday"),
    ("project proposal", "Tuesday", "Thursday"),
    ("lab report", "March 3", "March 10"),
    ("quiz correction", "week 4", "week 6"),
]

DEADLINES_KO = [
    ("과제", "월요일", "금요일"),
    ("프로젝트 제안서", "화요일", "목요일"),
    ("실험 보고서", "3월 3일", "3월 10일"),
    ("퀴즈 정정본", "4주차", "6주차"),
]

SCOPES_EN = [
    ("midterm exam", "chapters 1 through 3", "chapters 4 through 6"),
    ("final exam", "dynamic programming", "graph traversal"),
    ("quiz", "process scheduling", "virtual memory"),
    ("presentation", "retrieval augmented generation", "prompt engineering"),
]

SCOPES_KO = [
    ("중간고사", "1장부터 3장까지", "4장부터 6장까지"),
    ("기말고사", "동적 계획법", "그래프 탐색"),
    ("퀴즈", "프로세스 스케줄링", "가상 메모리"),
    ("발표", "검색 증강 생성", "프롬프트 엔지니어링"),
]

DEFINITIONS_EN = [
    ("process", "a program in execution", "a lightweight execution flow inside a process"),
    ("thread", "a lightweight execution flow inside a process", "a program in execution"),
    ("cache hit", "finding requested data in cache", "failing to find requested data in cache"),
    ("deadlock", "tasks waiting forever for each other", "a temporary delay caused by scheduling"),
]

DEFINITIONS_KO = [
    ("프로세스", "실행 중인 프로그램", "프로세스 안의 가벼운 실행 흐름"),
    ("스레드", "프로세스 안의 가벼운 실행 흐름", "실행 중인 프로그램"),
    ("캐시 히트", "요청한 데이터를 캐시에서 찾는 것", "요청한 데이터를 캐시에서 찾지 못하는 것"),
    ("교착상태", "작업들이 서로를 기다리며 영원히 멈춘 상태", "스케줄링 때문에 생기는 일시적 지연"),
]

COMPARISONS_EN = [
    ("merge sort", "quick sort", "more stable"),
    ("BFS", "DFS", "better for shortest paths in an unweighted graph"),
    ("SSD", "HDD", "faster for random access"),
    ("TCP", "UDP", "more reliable"),
]

COMPARISONS_KO = [
    ("병합 정렬", "퀵 정렬", "더 안정적인"),
    ("BFS 방식", "DFS 방식", "가중치가 없는 그래프의 최단 경로에 더 적합한"),
    ("SSD 저장장치", "HDD 저장장치", "임의 접근이 더 빠른"),
    ("TCP 프로토콜", "UDP 프로토콜", "더 신뢰성 있는"),
]

CODEWORDS_EN = [
    ("nulpax", "tovren", "mavik", "seldor"),
    ("brindle", "kavon", "lumer", "fandis"),
    ("orvex", "melpa", "zinter", "calnor"),
    ("talmin", "prexol", "daskar", "winrel"),
    ("quorbi", "sivren", "paldor", "nexil"),
]

CODEWORDS_KO = [
    ("라멜", "소핀", "루반", "가딘"),
    ("도르민", "카엘", "미루", "세폰"),
    ("누벡", "파린", "로딘", "하벨"),
    ("자민", "코렌", "비도", "수란"),
    ("테바", "모린", "가론", "리펜"),
]

ROOMS_EN = [
    ("review session", "Room N-204", "Room H-101"),
    ("lab meeting", "Building A 302", "Building C 501"),
    ("office hours", "Room S-210", "Room E-330"),
    ("group presentation", "Lecture Hall B", "Seminar Room 3"),
]

ROOMS_KO = [
    ("보강 수업", "새빛관 204호", "해오름관 101호"),
    ("실습 모임", "A동 302호", "C동 501호"),
    ("상담 시간", "S관 210호", "E관 330호"),
    ("조별 발표", "대강당 B", "세미나실 3"),
]

SCORES_EN = [
    ("attendance", "10 points", "15 points"),
    ("midterm exam", "30 points", "35 points"),
    ("final project", "40 points", "50 points"),
    ("time limit for the quiz", "30 minutes", "45 minutes"),
]

SCORES_KO = [
    ("출석", "10점", "15점"),
    ("중간고사", "30점", "35점"),
    ("기말 프로젝트", "40점", "50점"),
    ("퀴즈 제한 시간", "30분", "45분"),
]

POLICIES_EN = [
    ("late submission", "accepted with 10% penalty", "not accepted"),
    ("resubmission", "allowed once", "not allowed"),
    ("attendance", "excused with medical certificate", "excused with prior notification"),
    ("group size", "3 to 4 members", "2 to 3 members"),
]

POLICIES_KO = [
    ("지각 제출", "10% 감점으로 허용", "허용 안 됨"),
    ("재제출", "1회 허용", "허용 안 됨"),
    ("결석", "진단서 제출 시 인정", "사전 통보 시 인정"),
    ("조 크기", "3명에서 4명", "2명에서 3명"),
]

ASSIGNMENTS_EN = [
    ("dataset collection", "Team Alpha", "Team Beta"),
    ("code review", "Alice", "Bob"),
    ("presentation prep", "Charlie", "David"),
    ("documentation", "Emily", "Frank"),
]

ASSIGNMENTS_KO = [
    ("데이터셋 수집", "알파팀", "베타팀"),
    ("코드 리뷰", "민수", "지영"),
    ("발표 준비", "현우", "수진"),
    ("문서 작성", "태희", "준호"),
]

SPEAKERS_EN = ["Professor Lee", "TA Mina", "Instructor Park"]
SPEAKERS_KO = ["이 교수", "민아 조교", "박 강사"]


def has_hangul(text: str) -> bool:
    return bool(HANGUL_RE.search(str(text or "")))


def add_pair(rows: list[dict], *, source_id: str, question: str, passage_a: str, answer_a: str, passage_b: str, answer_b: str, split: str) -> None:
    contrast_id = f"{source_id}:{question}"
    rows.append({
        "source_id": f"{source_id}:a",
        "task": "final_qa",
        "split": split,
        "question": question,
        "answer": answer_a,
        "passage": passage_a,
        "contrast_id": contrast_id,
        "hard_negatives": [{"passage": passage_b, "answer": answer_b}],
    })
    rows.append({
        "source_id": f"{source_id}:b",
        "task": "final_qa",
        "split": split,
        "question": question,
        "answer": answer_b,
        "passage": passage_b,
        "contrast_id": contrast_id,
        "hard_negatives": [{"passage": passage_a, "answer": answer_a}],
    })


def build_rows() -> list[dict]:
    rows: list[dict] = []
    add_pair(
        rows,
        source_id="en_shared_diagnostic_homework_due",
        question=SERVICE_DIAGNOSTIC_CASE["question"],
        passage_a=SERVICE_DIAGNOSTIC_CASE["passage"],
        answer_a=SERVICE_DIAGNOSTIC_CASE["answer"],
        passage_b=SERVICE_DIAGNOSTIC_CASE["compare_passage"],
        answer_b=SERVICE_DIAGNOSTIC_CASE["compare_answer"],
        split="train",
    )
    add_pair(
        rows,
        source_id="en_shared_diagnostic_proposal_due",
        question=SERVICE_DIAGNOSTIC_CASE["alt_question"],
        passage_a=SERVICE_DIAGNOSTIC_CASE["passage"],
        answer_a=SERVICE_DIAGNOSTIC_CASE["alt_answer"],
        passage_b=SERVICE_DIAGNOSTIC_CASE["compare_passage"],
        answer_b=SERVICE_DIAGNOSTIC_CASE["answer"],
        split="train",
    )

    for i, (winner, loser) in enumerate(EN_TEAMS):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 5 == 4 else "train"
        pa = (
            f"{speaker}: {winner} defeated {loser} 3-1 in yesterday's class tournament. "
            f"The winner was {winner}. The loser was {loser}."
        )
        pb = (
            f"{speaker}: {loser} defeated {winner} 3-1 in yesterday's class tournament. "
            f"The winner was {loser}. The loser was {winner}."
        )
        add_pair(rows, source_id=f"en_match_winner_{i}", question="Who won the match?", passage_a=pa, answer_a=winner, passage_b=pb, answer_b=loser, split=split)
        add_pair(rows, source_id=f"en_match_loser_{i}", question="Which team lost the match?", passage_a=pa, answer_a=loser, passage_b=pb, answer_b=winner, split=split)

    for i, (winner, loser) in enumerate(KO_TEAMS):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 5 == 4 else "train"
        pa = f"{speaker}: 어제 수업 경기 결과는 {winner} 승리, {loser} 패배였습니다. 승자는 {winner}입니다."
        pb = f"{speaker}: 어제 수업 경기 결과는 {loser} 승리, {winner} 패배였습니다. 승자는 {loser}입니다."
        add_pair(rows, source_id=f"ko_match_winner_{i}", question="경기에서 누가 이겼어?", passage_a=pa, answer_a=winner, passage_b=pb, answer_b=loser, split=split)
        add_pair(rows, source_id=f"ko_match_loser_{i}", question="경기에서 진 팀은 어디야?", passage_a=pa, answer_a=loser, passage_b=pb, answer_b=winner, split=split)

    for i, (item, first, second) in enumerate(DEADLINES_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: The {item} is due on {first}. Please submit it before class."
        pb = f"{speaker}: The {item} is due on {second}. Please submit it before class."
        add_pair(rows, source_id=f"en_deadline_{i}", question=f"When is the {item} due?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (item, first, second) in enumerate(DEADLINES_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {item} 제출 기한은 {first}입니다. 수업 전에 제출하세요."
        pb = f"{speaker}: {item} 제출 기한은 {second}입니다. 수업 전에 제출하세요."
        add_pair(rows, source_id=f"ko_deadline_{i}", question=f"{item} 제출 기한은 언제야?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (exam, first, second) in enumerate(SCOPES_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: The scope of the {exam} is {first}. Other topics are not included."
        pb = f"{speaker}: The scope of the {exam} is {second}. Other topics are not included."
        add_pair(rows, source_id=f"en_scope_{i}", question=f"What is the scope of the {exam}?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (exam, first, second) in enumerate(SCOPES_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {exam} 범위는 {first}입니다. 다른 주제는 포함하지 않습니다."
        pb = f"{speaker}: {exam} 범위는 {second}입니다. 다른 주제는 포함하지 않습니다."
        add_pair(rows, source_id=f"ko_scope_{i}", question=f"{exam} 범위가 뭐야?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (term, first, second) in enumerate(DEFINITIONS_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: In today's lecture, {term} means {first}."
        pb = f"{speaker}: In today's lecture, {term} means {second}."
        add_pair(rows, source_id=f"en_definition_{i}", question=f"What did the instructor say {term} means?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (term, first, second) in enumerate(DEFINITIONS_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: 오늘 수업에서 {term}은 {first}이라고 설명했습니다."
        pb = f"{speaker}: 오늘 수업에서 {term}은 {second}이라고 설명했습니다."
        add_pair(rows, source_id=f"ko_definition_{i}", question=f"{term}을 뭐라고 설명했어?", passage_a=pa, answer_a=first, passage_b=pb, answer_b=second, split=split)

    for i, (left, right, relation) in enumerate(COMPARISONS_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: Compared with {right}, {left} is {relation}."
        pb = f"{speaker}: Compared with {left}, {right} is {relation}."
        add_pair(rows, source_id=f"en_compare_{i}", question=f"Which one is {relation}?", passage_a=pa, answer_a=left, passage_b=pb, answer_b=right, split=split)

    for i, (left, right, relation) in enumerate(COMPARISONS_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {right}와 비교했을 때 {left} 쪽이 {relation} 선택지입니다."
        pb = f"{speaker}: {left}와 비교했을 때 {right} 쪽이 {relation} 선택지입니다."
        add_pair(rows, source_id=f"ko_compare_{i}", question=f"{relation} 선택지는 무엇이야?", passage_a=pa, answer_a=left, passage_b=pb, answer_b=right, split=split)

    for i, (marker_a, marker_b, code_a, code_b) in enumerate(CODEWORDS_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 5 == 4 else "train"
        pa = (
            f"{speaker}: Private ledger Q-{i} says the {marker_a} marker is assigned "
            f"the code word {code_a}. The {marker_b} marker is assigned the code word {code_b}."
        )
        pb = (
            f"{speaker}: Private ledger Q-{i} says the {marker_a} marker is assigned "
            f"the code word {code_b}. The {marker_b} marker is assigned the code word {code_a}."
        )
        add_pair(
            rows,
            source_id=f"en_codeword_{i}",
            question=f"What code word is assigned to the {marker_a} marker?",
            passage_a=pa,
            answer_a=code_a,
            passage_b=pb,
            answer_b=code_b,
            split=split,
        )
        add_pair(
            rows,
            source_id=f"en_codeword_alt_{i}",
            question=f"What code word is assigned to the {marker_b} marker?",
            passage_a=pa,
            answer_a=code_b,
            passage_b=pb,
            answer_b=code_a,
            split=split,
        )

    for i, (marker_a, marker_b, code_a, code_b) in enumerate(CODEWORDS_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 5 == 4 else "train"
        pa = (
            f"{speaker}: 비공개 기록 Q-{i}에는 {marker_a} 표식의 암호어가 {code_a}라고 적혀 있습니다. "
            f"{marker_b} 표식의 암호어는 {code_b}입니다."
        )
        pb = (
            f"{speaker}: 비공개 기록 Q-{i}에는 {marker_a} 표식의 암호어가 {code_b}라고 적혀 있습니다. "
            f"{marker_b} 표식의 암호어는 {code_a}입니다."
        )
        add_pair(
            rows,
            source_id=f"ko_codeword_{i}",
            question=f"{marker_a} 표식에 배정된 암호어는 뭐야?",
            passage_a=pa,
            answer_a=code_a,
            passage_b=pb,
            answer_b=code_b,
            split=split,
        )
        add_pair(
            rows,
            source_id=f"ko_codeword_alt_{i}",
            question=f"{marker_b} 표식에 배정된 암호어는 뭐야?",
            passage_a=pa,
            answer_a=code_b,
            passage_b=pb,
            answer_b=code_a,
            split=split,
        )

    for i, (event, room_a, room_b) in enumerate(ROOMS_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: The {event} is assigned to {room_a}. Please check the updated schedule."
        pb = f"{speaker}: The {event} is assigned to {room_b}. Please check the updated schedule."
        add_pair(rows, source_id=f"en_room_{i}", question=f"Which room is assigned to the {event}?", passage_a=pa, answer_a=room_a, passage_b=pb, answer_b=room_b, split=split)

    for i, (event, room_a, room_b) in enumerate(ROOMS_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {event} 장소는 {room_a}입니다. 변경된 일정을 확인하세요."
        pb = f"{speaker}: {event} 장소는 {room_b}입니다. 변경된 일정을 확인하세요."
        add_pair(rows, source_id=f"ko_room_{i}", question=f"{event} 장소는 어디야?", passage_a=pa, answer_a=room_a, passage_b=pb, answer_b=room_b, split=split)

    for i, (item, score_a, score_b) in enumerate(SCORES_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: The weight for {item} is {score_a}. This is different from last semester."
        pb = f"{speaker}: The weight for {item} is {score_b}. This is different from last semester."
        add_pair(rows, source_id=f"en_score_{i}", question=f"What is the weight for {item}?", passage_a=pa, answer_a=score_a, passage_b=pb, answer_b=score_b, split=split)

    for i, (item, score_a, score_b) in enumerate(SCORES_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {item} 배점은 {score_a}입니다. 지난 학기와 다릅니다."
        pb = f"{speaker}: {item} 배점은 {score_b}입니다. 지난 학기와 다릅니다."
        add_pair(rows, source_id=f"ko_score_{i}", question=f"{item} 배점은 얼마야?", passage_a=pa, answer_a=score_a, passage_b=pb, answer_b=score_b, split=split)

    for i, (policy, rule_a, rule_b) in enumerate(POLICIES_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: The policy for {policy} is {rule_a}. Please note this carefully."
        pb = f"{speaker}: The policy for {policy} is {rule_b}. Please note this carefully."
        add_pair(rows, source_id=f"en_policy_{i}", question=f"What is the policy for {policy}?", passage_a=pa, answer_a=rule_a, passage_b=pb, answer_b=rule_b, split=split)

    for i, (policy, rule_a, rule_b) in enumerate(POLICIES_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {policy} 정책은 {rule_a}입니다. 주의해서 확인하세요."
        pb = f"{speaker}: {policy} 정책은 {rule_b}입니다. 주의해서 확인하세요."
        add_pair(rows, source_id=f"ko_policy_{i}", question=f"{policy} 정책이 뭐야?", passage_a=pa, answer_a=rule_a, passage_b=pb, answer_b=rule_b, split=split)

    for i, (task, person_a, person_b) in enumerate(ASSIGNMENTS_EN):
        speaker = SPEAKERS_EN[i % len(SPEAKERS_EN)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {person_a} is assigned to {task}. {person_b} is assigned to another task."
        pb = f"{speaker}: {person_b} is assigned to {task}. {person_a} is assigned to another task."
        add_pair(rows, source_id=f"en_assignment_{i}", question=f"Who is assigned to {task}?", passage_a=pa, answer_a=person_a, passage_b=pb, answer_b=person_b, split=split)

    for i, (task, person_a, person_b) in enumerate(ASSIGNMENTS_KO):
        speaker = SPEAKERS_KO[i % len(SPEAKERS_KO)]
        split = "valid" if i % 4 == 3 else "train"
        pa = f"{speaker}: {task} 담당은 {person_a}입니다. {person_b}는 다른 과제를 맡았습니다."
        pb = f"{speaker}: {task} 담당은 {person_b}입니다. {person_a}는 다른 과제를 맡았습니다."
        add_pair(rows, source_id=f"ko_assignment_{i}", question=f"{task} 담당이 누구야?", passage_a=pa, answer_a=person_a, passage_b=pb, answer_b=person_b, split=split)

    return rows


def expand_rows(rows: list[dict], repeats: int) -> list[dict]:
    if repeats <= 1:
        return rows
    expanded = []
    for repeat_idx in range(repeats):
        session = repeat_idx + 1
        for row in rows:
            cloned = dict(row)
            prefix = f"수업 {session}회차. " if has_hangul(row["question"]) else f"Session {session}. "
            cloned["source_id"] = f"{row['source_id']}:session{session:02d}"
            cloned["contrast_id"] = f"{row['contrast_id']}:session{session:02d}"
            cloned["lecture_id"] = f"synthetic-service-{session:02d}"
            cloned["passage"] = f"{prefix}{row['passage']}"
            cloned["hard_negatives"] = [
                {
                    **negative,
                    "passage": f"{prefix}{negative['passage']}",
                }
                for negative in row.get("hard_negatives", [])
            ]
            expanded.append(cloned)
    return expanded


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {k: v for k, v in row.items() if k != "split"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-name", default="ServiceHardPair_train.jsonl")
    parser.add_argument("--valid-name", default="ServiceHardPair_valid.jsonl")
    parser.add_argument("--train-repeats", type=int, default=30)
    parser.add_argument("--valid-repeats", type=int, default=5)
    args = parser.parse_args()

    rows = build_rows()
    train_rows = expand_rows([row for row in rows if row["split"] == "train"], args.train_repeats)
    valid_rows = expand_rows([row for row in rows if row["split"] == "valid"], args.valid_repeats)

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_name
    valid_path = output_dir / args.valid_name
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    print(f"[service_hardpairs] train={len(train_rows)} -> {train_path}")
    print(f"[service_hardpairs] valid={len(valid_rows)} -> {valid_path}")


if __name__ == "__main__":
    main()
