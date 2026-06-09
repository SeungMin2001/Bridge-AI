#!/usr/bin/env python3
"""Smoke-check demo-critical quality gates without starting the web servers.

This script intentionally focuses on deterministic guards that protect the
presentation path: schedule extraction context, quiz fallback quality, and
summary structure. LLM-backed generation still needs a live demo check, but
these checks catch the regressions that previously broke the demo.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class SmokeFailure(AssertionError):
    """Raised when a demo-critical invariant is violated."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        print(f"[skip] {name}: missing dependency ({exc.name})")
        return None


def check_schedule_quality() -> None:
    schedule_service = _load_module("schedule.schedule_service")
    if schedule_service is None:
        return

    cases = [
        {
            "name": "exam_detail_context",
            "text": "저희 기말고사는 1월 3일에 봅니다. 딥러닝 단원 준비해주세요.",
            "must_title": ("기말고사", "딥러닝"),
            "must_source": ("1월 3일", "딥러닝 단원"),
            "date": "-01-03T09:00:00",
            "count": 1,
        },
        {
            "name": "assignment_topic_context",
            "text": "이번 과제는 신경망 역전파 보고서입니다. 제출은 6월 14일까지 해주세요.",
            "must_title": ("신경망", "역전파", "과제"),
            "must_source": ("신경망 역전파", "6월 14일"),
            "date": "-06-14T23:59:00",
            "count": 1,
        },
        {
            "name": "presentation_detail_context",
            "text": "다음 주 화요일 오후 2시에 최종 발표가 있습니다. 발표 주제는 BridgePRAG 서비스 시연입니다.",
            "must_title": ("최종 발표", "BridgePRAG"),
            "must_source": ("다음 주 화요일", "BridgePRAG 서비스 시연"),
            "date_contains": "T14:00:00",
            "count": 1,
        },
        {
            "name": "ignore_personal_schedule",
            "text": "내일 친구랑 밥 먹기로 했고, 6월 20일에는 여행을 갑니다.",
            "count": 0,
        },
    ]

    for case in cases:
        schedules = schedule_service._extract_rule_based_schedules(case["text"])
        _assert(
            len(schedules) == case["count"],
            f"[schedule:{case['name']}] expected {case['count']} item(s), got {len(schedules)}: {schedules}",
        )
        if not schedules:
            continue
        item = schedules[0]
        title = _compact(item.get("title"))
        source = _compact(item.get("source_text"))
        due_date = str(item.get("due_date") or "")
        for token in case.get("must_title", ()):
            _assert(token in title, f"[schedule:{case['name']}] title missing '{token}': {title}")
        for token in case.get("must_source", ()):
            _assert(token in source, f"[schedule:{case['name']}] source missing '{token}': {source}")
        if case.get("date"):
            _assert(case["date"] in due_date, f"[schedule:{case['name']}] bad date: {due_date}")
        if case.get("date_contains"):
            _assert(case["date_contains"] in due_date, f"[schedule:{case['name']}] bad time: {due_date}")

    print("[ok] schedule quality")


def check_quiz_quality() -> None:
    quiz_service = _load_module("quiz.quiz_service")
    if quiz_service is None:
        return

    source_text = """
    인공신경망은 입력층, 은닉층, 출력층으로 구성됩니다. 입력층은 데이터를 받아들이고,
    은닉층은 가중치와 활성화 함수를 통해 패턴을 표현합니다. 출력층은 최종 예측값을 만듭니다.
    손실 함수는 모델의 예측값과 정답 사이의 차이를 수치로 나타내는 기준입니다.
    회귀 문제에서는 평균제곱오차를 사용할 수 있고, 분류 문제에서는 교차 엔트로피 손실을 자주 사용합니다.
    역전파는 출력층에서 계산된 손실을 기준으로 각 층의 가중치가 손실에 얼마나 영향을 주었는지 계산하는 과정입니다.
    경사하강법은 손실이 줄어드는 방향으로 가중치를 조금씩 갱신합니다. 학습률은 한 번에 가중치를 얼마나 바꿀지 정하는 값입니다.
    학습률이 너무 크면 최적점을 지나칠 수 있고, 너무 작으면 학습이 매우 느려질 수 있습니다.
    과적합은 훈련 데이터에는 잘 맞지만 새로운 데이터에는 성능이 낮아지는 현상입니다.
    드롭아웃은 일부 뉴런을 임시로 제외해 과적합을 줄이는 정규화 방법입니다.
    """
    expected_counts = {"MULTIPLE_CHOICE": 5, "OX": 3}
    quiz_data = quiz_service._fit_quiz_to_expected_counts([], expected_counts, source_text)
    issues = quiz_service._validate_quiz_quality(quiz_data, expected_counts)
    _assert(not issues, "[quiz] fallback quality issues: " + "; ".join(issues[:5]))

    seen_questions: set[str] = set()
    forbidden = (
        "...",
        "자료에서 확인할 수 없는",
        "자료의 설명과 반대",
        "관련이 있다",
        "관련이 없다",
        "하나의 고정값으로만",
    )
    for index, question in enumerate(quiz_data, start=1):
        question_text = _compact(question.get("question"))
        explanation = _compact(question.get("explanation"))
        _assert(question_text and question_text not in seen_questions, f"[quiz] duplicate/empty question #{index}")
        seen_questions.add(question_text)
        _assert(explanation.endswith((".", "!", "?")), f"[quiz] incomplete explanation #{index}: {explanation}")
        _assert("..." not in explanation and "…" not in explanation, f"[quiz] ellipsis explanation #{index}: {explanation}")
        if question.get("type") == "MULTIPLE_CHOICE":
            for option in question.get("options") or []:
                option_text = _compact(option)
                _assert(
                    not any(token in option_text for token in forbidden),
                    f"[quiz] weak option #{index}: {option_text}",
                )

    repaired = quiz_service._complete_explanation("손실 함수는 예측값과 정답의 차이를 나타내는 기준입니...")
    _assert(repaired.endswith("기준입니다."), f"[quiz] explanation repair failed: {repaired}")
    print("[ok] quiz quality")


def check_summary_quality() -> None:
    summary_service = _load_module("summary.summary_service")
    if summary_service is None:
        return

    source_text = """
    손실 함수는 모델의 예측값과 정답 사이의 차이를 수치로 나타내는 기준입니다.
    회귀 문제에서는 평균제곱오차를 사용할 수 있고, 분류 문제에서는 교차 엔트로피 손실을 자주 사용합니다.
    역전파는 출력층에서 계산된 손실을 기준으로 각 층의 가중치가 손실에 얼마나 영향을 주었는지 계산하는 과정입니다.
    경사하강법은 손실이 줄어드는 방향으로 가중치를 조금씩 갱신합니다.
    학습률은 한 번에 가중치를 얼마나 바꿀지 정하는 값이며, 너무 크면 최적점을 지나칠 수 있고 너무 작으면 학습이 느려집니다.
    """
    weak_summary = "## 핵심 요약\n보고서는 전체 흐름과 세부 설명을 포함합니다.\n## 주요 내용\n학습 포인트를 명확하게 정리했습니다."
    summary = summary_service._ensure_structured_summary(weak_summary, source_text)
    for section in ("## 핵심 요약", "## 주요 내용", "## 학습 포인트"):
        _assert(section in summary, f"[summary] missing section: {section}")
    for concept in ("손실 함수", "역전파", "학습률"):
        _assert(concept in summary, f"[summary] missing source concept: {concept}")
    _assert("보고서는 전체 흐름" not in summary, "[summary] generic filler survived")
    print("[ok] summary quality")


def main() -> int:
    checks = (
        check_schedule_quality,
        check_quiz_quality,
        check_summary_quality,
    )
    failed = False
    for check in checks:
        try:
            check()
        except SmokeFailure as exc:
            failed = True
            print(f"[fail] {exc}")
        except Exception as exc:  # Keep demo smoke output actionable.
            failed = True
            print(f"[error] {check.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        return 1
    print("[ok] demo quality smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
