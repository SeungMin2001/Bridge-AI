#!/usr/bin/env python3
"""Seed BridgePRAG-style lecture transcript samples into the local workspace DB.

The service does not store QA pairs directly. Instead, transcript chunks act as
retrieved passages and user questions act as QA queries. This script inserts
lecture-like passages with clear multi-fact content and prints suggested test
questions that should retrieve those passages.
"""

from __future__ import annotations

import json
import sys
import uuid
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_batch, register_uuid


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db_config import psycopg2_config  # noqa: E402


register_uuid()

FOLDER_TITLE = "BridgePRAG 샘플 강의"
FOLDER_DESCRIPTION = "BridgePRAG 서비스 테스트용 강의형 샘플 데이터"
SAMPLE_PREFIX = "bridgeprag-lecture-sample"
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


@dataclass(frozen=True)
class LectureSample:
    slug: str
    session_title: str
    recording_title: str
    course_title: str
    chunks: tuple[str, ...]
    questions: tuple[str, ...]


SAMPLES: tuple[LectureSample, ...] = (
    LectureSample(
        slug="os-process-thread",
        course_title="운영체제",
        session_title="운영체제 3주차 - 프로세스와 스레드",
        recording_title="운영체제 3주차 프로세스와 스레드 녹음본",
        chunks=(
            "오늘은 프로세스와 스레드의 차이를 정리하겠습니다. 프로세스는 실행 중인 프로그램의 단위이고, 독립적인 주소 공간과 자원을 가집니다. 반면 스레드는 하나의 프로세스 안에서 실행되는 흐름이기 때문에 코드 영역과 힙 영역을 공유하지만, 각 스레드는 자기만의 스택과 레지스터 상태를 가집니다.",
            "문맥 교환 비용은 프로세스보다 스레드가 일반적으로 더 작습니다. 프로세스 문맥 교환은 주소 공간과 페이지 테이블 전환까지 포함될 수 있지만, 스레드 문맥 교환은 같은 주소 공간 안에서 실행 흐름만 바뀌는 경우가 많기 때문입니다. 그래서 웹 서버처럼 동시에 많은 요청을 처리하는 경우 스레드를 활용하면 응답성이 좋아질 수 있습니다.",
            "다만 스레드가 메모리를 공유한다는 장점은 동시에 위험이 됩니다. 여러 스레드가 같은 변수에 동시에 접근하면 실행 순서에 따라 결과가 달라지는 경쟁 상태가 발생할 수 있습니다. 이 문제를 줄이기 위해 임계 구역에는 뮤텍스나 세마포어 같은 동기화 기법을 적용합니다.",
            "임계 구역 문제를 해결할 때는 세 가지 조건을 확인해야 합니다. 첫째, 한 번에 하나의 스레드만 들어가는 상호 배제 조건이 필요합니다. 둘째, 들어갈 수 있는 스레드가 있다면 선택이 무한히 지연되지 않는 진행 조건이 필요합니다. 셋째, 특정 스레드가 계속 밀리지 않도록 대기 횟수를 제한하는 한정 대기 조건이 필요합니다.",
        ),
        questions=(
            "프로세스와 스레드의 차이를 강의에서는 어떻게 설명했어?",
            "스레드에서 경쟁 상태가 생기는 이유와 해결 방법은 뭐야?",
            "임계 구역 문제의 세 가지 조건은 무엇이야?",
        ),
    ),
    LectureSample(
        slug="db-index-transaction",
        course_title="데이터베이스",
        session_title="데이터베이스 5주차 - 인덱스와 트랜잭션",
        recording_title="데이터베이스 5주차 인덱스와 트랜잭션 녹음본",
        chunks=(
            "이번 시간에는 인덱스와 트랜잭션을 함께 보겠습니다. 인덱스는 책의 색인처럼 원하는 행을 빨리 찾기 위한 보조 자료구조입니다. 데이터베이스에서는 B+트리 인덱스가 자주 사용되며, 루트에서 리프 노드까지 이동하면서 검색 범위를 점점 좁혀 갑니다.",
            "복합 인덱스에서는 컬럼 순서가 중요합니다. 예를 들어 학생 테이블에 학과와 학번 순서로 인덱스를 만들면, 학과 조건으로 먼저 좁힌 뒤 학번을 찾는 질의에는 효과적입니다. 하지만 학번만 단독으로 찾는 질의에서는 인덱스의 앞쪽 컬럼을 건너뛰기 어렵기 때문에 효과가 줄어들 수 있습니다.",
            "트랜잭션은 데이터베이스에서 하나의 논리적 작업 단위입니다. 원자성은 작업이 모두 성공하거나 모두 취소되어야 한다는 의미이고, 일관성은 트랜잭션 전후에 제약 조건이 깨지지 않아야 한다는 의미입니다. 고립성은 동시에 실행되는 트랜잭션이 서로 간섭하지 않도록 보장하는 성질이고, 지속성은 커밋된 결과가 장애 이후에도 남아야 한다는 성질입니다.",
            "장애 복구를 위해 로그 선행 기록, 즉 Write-Ahead Logging을 사용합니다. 데이터 페이지를 디스크에 쓰기 전에 변경 내용에 대한 로그를 먼저 안정적인 저장소에 기록합니다. 이렇게 하면 시스템이 중간에 멈춰도 로그를 기준으로 redo와 undo를 수행해 커밋된 결과는 살리고 미완료 작업은 되돌릴 수 있습니다.",
        ),
        questions=(
            "B+트리 인덱스는 강의에서 어떤 방식으로 설명됐어?",
            "복합 인덱스에서 컬럼 순서가 중요한 이유는 뭐야?",
            "트랜잭션의 ACID 속성을 설명해줘.",
        ),
    ),
    LectureSample(
        slug="rag-memory-bridge",
        course_title="인공지능 응용",
        session_title="RAG 7주차 - 검색 증강 생성과 메모리",
        recording_title="RAG 7주차 검색 증강 생성과 메모리 녹음본",
        chunks=(
            "검색 증강 생성은 대규모 언어모델이 모든 지식을 파라미터 안에 외우고 있다고 가정하지 않습니다. 사용자의 질문이 들어오면 관련 문서를 먼저 검색하고, 검색된 문맥을 모델에게 함께 제공해 답변을 생성합니다. 이 방식은 최신 정보나 수업 자료처럼 모델이 사전학습에서 보지 못한 지식을 다룰 때 유용합니다.",
            "검색된 passage의 수를 늘리면 답을 찾을 가능성은 높아질 수 있지만, 동시에 모델이 읽어야 하는 문맥 길이가 길어지고 노이즈 passage가 섞일 수 있습니다. 따라서 top-k를 무조건 크게 잡는 것보다 질문과 직접 관련된 passage를 고르고, 필요하면 재랭킹을 통해 근거 품질을 높이는 것이 중요합니다.",
            "Parametric RAG 계열 접근은 외부 지식을 긴 텍스트로만 넣지 않고, 모델 내부에서 사용할 수 있는 메모리 표현으로 바꾸려는 방향입니다. 예를 들어 passage를 K/V 메모리로 변환하면 어텐션 계산 과정에서 해당 정보를 참조하게 만들 수 있습니다. 이때 여러 passage의 정보가 충돌하지 않도록 병합 방식도 함께 고려해야 합니다.",
            "BridgePRAG의 핵심 아이디어는 passage만 따로 인코딩하지 않고 질문과 passage를 함께 묶어 메모리를 만드는 것입니다. 같은 passage라도 질문이 요구하는 정보가 다르면 중요하게 봐야 할 부분이 달라질 수 있습니다. 그래서 질문-문맥 pair를 기반으로 K/V 메모리를 만들면, 현재 질문 의도에 맞는 정보가 더 잘 반영될 수 있습니다.",
        ),
        questions=(
            "RAG에서 passage 수를 늘릴 때 생기는 장점과 단점은 뭐야?",
            "Parametric RAG는 외부 지식을 어떻게 활용하려는 접근이야?",
            "BridgePRAG가 passage만 인코딩하지 않고 질문과 passage를 함께 쓰는 이유는 뭐야?",
        ),
    ),
    LectureSample(
        slug="deep-learning-week1-neural-network",
        course_title="딥러닝",
        session_title="딥러닝 1주차 - 신경망과 역전파",
        recording_title="딥러닝 1주차 신경망과 역전파 녹음본",
        chunks=(
            "딥러닝 1주차에서는 인공신경망의 기본 구조를 다룹니다. 뉴런은 입력값에 가중치를 곱해 더한 뒤 활성화 함수를 통과시켜 출력을 만듭니다. 여러 뉴런을 층으로 쌓으면 입력층, 은닉층, 출력층으로 구성된 신경망이 되고, 은닉층이 깊어질수록 데이터의 복잡한 패턴을 단계적으로 표현할 수 있습니다.",
            "손실 함수는 모델의 예측값과 정답 사이의 차이를 수치로 나타내는 기준입니다. 회귀 문제에서는 평균제곱오차를 사용할 수 있고, 분류 문제에서는 교차 엔트로피 손실을 자주 사용합니다. 학습의 목표는 이 손실 값을 줄이는 방향으로 가중치를 조정하는 것입니다.",
            "역전파는 출력층에서 계산된 손실을 기준으로 각 층의 가중치가 손실에 얼마나 영향을 주었는지 계산하는 과정입니다. 체인 룰을 이용해 출력층에서 입력층 방향으로 기울기를 전달하고, 경사하강법은 이 기울기의 반대 방향으로 가중치를 조금씩 갱신합니다.",
            "학습률은 한 번의 업데이트에서 가중치를 얼마나 크게 바꿀지 결정합니다. 학습률이 너무 크면 손실이 발산하거나 최적점을 지나칠 수 있고, 너무 작으면 학습 속도가 지나치게 느려질 수 있습니다. 이번 주 과제는 간단한 다층 퍼셉트론을 구현하고 금요일 오후 여섯 시까지 제출하는 것입니다.",
        ),
        questions=(
            "딥러닝 1주차에서 뉴런과 신경망 구조를 어떻게 설명했어?",
            "역전파와 경사하강법의 관계는 뭐야?",
            "딥러닝 1주차 과제는 언제까지 제출해야 해?",
        ),
    ),
    LectureSample(
        slug="deep-learning-week2-cnn",
        course_title="딥러닝",
        session_title="딥러닝 2주차 - CNN과 특징 추출",
        recording_title="딥러닝 2주차 CNN과 특징 추출 녹음본",
        chunks=(
            "딥러닝 2주차에서는 합성곱 신경망, 즉 CNN을 다룹니다. CNN은 이미지 전체를 한 번에 완전연결층으로 처리하지 않고, 작은 필터를 이동시키며 지역적인 패턴을 추출합니다. 이 필터는 학습 가능한 가중치이며, 이미지의 모서리, 방향, 질감 같은 특징을 점차적으로 감지합니다.",
            "합성곱 연산에서 stride는 필터가 한 번에 이동하는 간격을 의미합니다. stride가 커지면 출력 feature map의 크기는 작아지고 계산량도 줄어듭니다. padding은 입력 이미지 가장자리에 값을 추가하는 방식이며, 출력 크기를 유지하거나 가장자리 정보 손실을 줄이기 위해 사용합니다.",
            "Pooling은 feature map의 공간 크기를 줄이면서 중요한 정보를 남기는 과정입니다. Max pooling은 영역 안에서 가장 큰 값을 선택하므로 강하게 반응한 특징을 보존하는 데 유용합니다. Average pooling은 영역의 평균값을 사용하므로 전체적인 분포를 부드럽게 반영합니다.",
            "CNN은 이미지 분류뿐만 아니라 객체 검출과 의료 영상 분석에도 활용됩니다. 낮은 층은 선이나 색 변화처럼 단순한 특징을 학습하고, 깊은 층은 눈, 바퀴, 건물 모서리처럼 더 추상적인 패턴을 학습합니다. 실습에서는 손글씨 숫자 데이터를 이용해 CNN 분류기를 훈련합니다.",
        ),
        questions=(
            "CNN에서 필터는 어떤 역할을 해?",
            "stride와 padding은 출력 크기에 어떤 영향을 줘?",
            "Max pooling과 Average pooling의 차이는 뭐야?",
        ),
    ),
    LectureSample(
        slug="deep-learning-week3-transformer",
        course_title="딥러닝",
        session_title="딥러닝 3주차 - Transformer와 Attention",
        recording_title="딥러닝 3주차 Transformer와 Attention 녹음본",
        chunks=(
            "딥러닝 3주차에서는 Transformer의 Attention 구조를 설명합니다. Attention은 현재 토큰이 문장 안의 다른 토큰 중 어떤 정보에 집중해야 하는지를 계산하는 방식입니다. Query는 현재 토큰이 찾고자 하는 정보의 기준이고, Key는 각 토큰이 가진 주소 역할을 하며, Value는 실제로 가져올 정보의 내용에 해당합니다.",
            "Scaled dot-product Attention은 Query와 Key의 내적을 계산해 관련도를 구한 뒤 softmax로 가중치를 만듭니다. 이 가중치를 Value에 곱해 더하면 현재 토큰이 참고해야 할 문맥 정보가 만들어집니다. 내적 값이 너무 커지는 것을 막기 위해 Key 차원의 제곱근으로 나누는 scaling을 적용합니다.",
            "Multi-head Attention은 하나의 Attention만 사용하는 대신 여러 개의 head를 병렬로 사용합니다. 각 head는 서로 다른 표현 공간에서 관계를 학습하므로 문법적 관계, 의미적 관계, 위치적 단서를 동시에 포착할 수 있습니다. 이후 head들의 출력을 연결하고 선형 변환하여 다음 층으로 전달합니다.",
            "Transformer는 순환 구조가 없기 때문에 토큰 순서를 별도로 알려주어야 합니다. 이를 위해 positional encoding을 입력 임베딩에 더합니다. 수업에서는 다음 시간까지 Query, Key, Value 행렬의 크기와 Attention 출력 크기를 계산하는 연습 문제를 풀어오라고 안내했습니다.",
        ),
        questions=(
            "Attention에서 Query, Key, Value는 각각 어떤 역할이야?",
            "Scaled dot-product Attention에서 scaling을 하는 이유는 뭐야?",
            "Transformer에서 positional encoding이 필요한 이유는 뭐야?",
        ),
    ),
    LectureSample(
        slug="economics-week1-supply-demand",
        course_title="경제학",
        session_title="경제학 1주차 - 수요와 공급",
        recording_title="경제학 1주차 수요와 공급 녹음본",
        chunks=(
            "경제학 1주차에서는 시장에서 가격이 어떻게 결정되는지 설명합니다. 수요는 소비자가 특정 가격에서 구매하려는 재화의 양이고, 공급은 생산자가 특정 가격에서 판매하려는 재화의 양입니다. 일반적으로 가격이 오르면 수요량은 줄고 공급량은 늘어나는 방향으로 움직입니다.",
            "수요곡선이 오른쪽으로 이동하는 경우는 소비자의 소득 증가, 선호 변화, 대체재 가격 상승처럼 같은 가격에서 더 많이 사고 싶어지는 상황입니다. 공급곡선이 오른쪽으로 이동하는 경우는 생산기술 향상이나 원자재 가격 하락처럼 같은 가격에서 더 많이 생산할 수 있는 상황입니다.",
            "균형가격은 수요량과 공급량이 일치하는 지점에서 결정됩니다. 가격이 균형보다 높으면 공급량이 수요량보다 많아 초과공급이 발생하고, 가격이 균형보다 낮으면 수요량이 공급량보다 많아 초과수요가 발생합니다. 시장은 이러한 압력을 통해 다시 균형으로 이동하려는 경향을 가집니다.",
            "정부가 가격상한제를 균형가격보다 낮게 설정하면 소비자는 더 많이 사려고 하지만 생산자는 덜 공급하려 하므로 부족 현상이 발생할 수 있습니다. 반대로 가격하한제가 균형가격보다 높게 설정되면 초과공급이 발생할 수 있습니다. 다음 수업 전까지 임대료 상한제 사례를 읽어오는 과제가 있습니다.",
        ),
        questions=(
            "수요와 공급은 각각 무엇을 의미해?",
            "균형가격보다 가격이 낮으면 어떤 현상이 생겨?",
            "가격상한제는 어떤 문제를 만들 수 있어?",
        ),
    ),
    LectureSample(
        slug="economics-week2-elasticity",
        course_title="경제학",
        session_title="경제학 2주차 - 탄력성과 소비자 선택",
        recording_title="경제학 2주차 탄력성과 소비자 선택 녹음본",
        chunks=(
            "경제학 2주차에서는 탄력성을 다룹니다. 가격탄력성은 가격이 1퍼센트 변할 때 수요량이 몇 퍼센트 변하는지를 나타냅니다. 절댓값이 1보다 크면 탄력적이라고 하고, 1보다 작으면 비탄력적이라고 합니다. 필수재는 보통 비탄력적이고, 대체재가 많은 상품은 탄력적인 경우가 많습니다.",
            "수요가 탄력적일 때 가격을 올리면 수요량이 크게 줄어 총수입이 감소할 수 있습니다. 반대로 수요가 비탄력적일 때 가격을 올리면 수요량 감소가 작기 때문에 총수입이 증가할 수 있습니다. 따라서 기업은 가격 전략을 세울 때 수요의 탄력성을 함께 고려해야 합니다.",
            "소비자 선택 이론에서는 예산 제약과 효용을 함께 봅니다. 예산선은 소비자가 주어진 소득과 가격에서 구매할 수 있는 상품 조합을 나타냅니다. 무차별곡선은 소비자에게 같은 만족을 주는 조합들의 집합이며, 예산선과 무차별곡선이 접하는 지점에서 최적 소비가 결정됩니다.",
            "이번 주 계산 과제는 가격탄력성과 총수입 변화를 표로 정리하는 것입니다. 제출 마감은 다음 주 월요일 오전 아홉 시입니다. 과제에서는 가격이 만 원에서 만천 원으로 오를 때 수요량이 백 개에서 팔십 개로 줄어드는 사례를 사용합니다.",
        ),
        questions=(
            "가격탄력성은 무엇을 나타내는 지표야?",
            "수요가 탄력적일 때 가격을 올리면 총수입은 어떻게 될 수 있어?",
            "경제학 2주차 과제 제출 마감은 언제야?",
        ),
    ),
    LectureSample(
        slug="economics-week3-market-failure",
        course_title="경제학",
        session_title="경제학 3주차 - 시장실패와 정부 개입",
        recording_title="경제학 3주차 시장실패와 정부 개입 녹음본",
        chunks=(
            "경제학 3주차에서는 시장실패를 설명합니다. 시장실패는 시장이 스스로 효율적인 자원 배분을 달성하지 못하는 상황을 의미합니다. 대표적인 원인에는 외부효과, 공공재, 정보 비대칭, 독점이 있습니다.",
            "외부효과는 한 경제 주체의 행동이 거래 당사자가 아닌 제삼자에게 영향을 주지만 그 영향이 가격에 충분히 반영되지 않는 경우입니다. 공장 배출가스는 주변 주민에게 피해를 주는 부정적 외부효과의 예입니다. 반대로 예방접종은 주변 사람의 감염 위험도 줄이는 긍정적 외부효과의 예입니다.",
            "공공재는 비배제성과 비경합성을 가진 재화입니다. 비배제성은 비용을 내지 않은 사람을 소비에서 배제하기 어렵다는 뜻이고, 비경합성은 한 사람이 소비해도 다른 사람의 소비 가능성이 줄어들지 않는다는 뜻입니다. 국방과 등대는 전통적인 공공재 사례로 설명됩니다.",
            "정부는 세금, 보조금, 규제, 공공재 직접 공급 등을 통해 시장실패를 완화하려고 합니다. 부정적 외부효과에는 피구세를 부과할 수 있고, 긍정적 외부효과에는 보조금을 지급할 수 있습니다. 다만 정부 개입도 정보 부족이나 행정 비용 때문에 항상 완벽한 결과를 보장하지는 않습니다.",
        ),
        questions=(
            "시장실패의 대표적인 원인은 무엇이야?",
            "부정적 외부효과와 긍정적 외부효과의 예시는 뭐야?",
            "공공재의 두 가지 성질은 무엇이야?",
        ),
    ),
    LectureSample(
        slug="physics-week1-motion-force",
        course_title="물리학",
        session_title="물리학 1주차 - 운동과 힘",
        recording_title="물리학 1주차 운동과 힘 녹음본",
        chunks=(
            "물리학 1주차에서는 운동을 기술하는 기본 물리량을 정리합니다. 위치는 물체가 기준점에서 어디에 있는지를 나타내고, 변위는 처음 위치에서 나중 위치까지의 방향을 가진 변화량입니다. 속도는 변위를 시간으로 나눈 값이고, 가속도는 속도가 시간에 따라 얼마나 변하는지를 나타냅니다.",
            "등가속도 직선 운동에서는 가속도가 일정하다고 가정합니다. 이때 속도는 시간에 비례해 변하고, 위치는 시간의 제곱 항을 포함해 변합니다. 자유낙하 운동은 공기저항을 무시하면 중력가속도 약 9.8미터 매초제곱을 갖는 등가속도 운동으로 다룰 수 있습니다.",
            "뉴턴의 제1법칙은 관성의 법칙입니다. 외부에서 알짜힘이 작용하지 않으면 물체는 정지 상태를 유지하거나 등속 직선 운동을 계속합니다. 제2법칙은 알짜힘이 질량과 가속도의 곱과 같다는 내용이며, 제3법칙은 작용과 반작용이 항상 크기가 같고 방향이 반대라는 내용입니다.",
            "수업에서는 힘의 단위를 뉴턴으로 정의했습니다. 1뉴턴은 질량 1킬로그램의 물체에 1미터 매초제곱의 가속도를 만들 때 필요한 힘입니다. 실습 문제는 물체에 작용하는 힘을 자유물체도로 나타내고 알짜힘을 계산하는 것입니다.",
        ),
        questions=(
            "속도와 가속도는 각각 무엇을 의미해?",
            "자유낙하 운동은 어떤 운동으로 설명했어?",
            "뉴턴의 세 가지 운동 법칙을 정리해줘.",
        ),
    ),
    LectureSample(
        slug="physics-week2-energy",
        course_title="물리학",
        session_title="물리학 2주차 - 일과 에너지",
        recording_title="물리학 2주차 일과 에너지 녹음본",
        chunks=(
            "물리학 2주차에서는 일과 에너지의 관계를 다룹니다. 물리에서 일은 힘이 물체를 이동시키면서 에너지를 전달하는 과정입니다. 힘과 이동 방향이 같을수록 일이 커지고, 힘이 이동 방향과 수직이면 일은 0이 됩니다.",
            "운동에너지는 물체가 운동하기 때문에 가지는 에너지입니다. 질량이 클수록, 속력이 클수록 운동에너지는 증가하며 특히 속력의 제곱에 비례합니다. 위치에너지는 중력장 안에서 높이에 의해 저장되는 에너지로, 질량과 중력가속도와 높이에 비례합니다.",
            "역학적 에너지 보존은 마찰이나 공기저항 같은 비보존력이 무시될 때 운동에너지와 위치에너지의 합이 일정하게 유지된다는 원리입니다. 롤러코스터가 높은 곳에서 내려올 때 위치에너지가 줄어드는 대신 운동에너지가 증가하는 사례로 설명할 수 있습니다.",
            "마찰이 존재하면 역학적 에너지 일부가 열에너지로 전환됩니다. 이 경우 운동에너지와 위치에너지의 합은 보존되지 않지만 전체 에너지는 다른 형태까지 포함하면 보존됩니다. 이번 주 실험 보고서는 경사면에서 구슬이 내려오는 시간을 측정하고 에너지 손실을 추정하는 내용입니다.",
        ),
        questions=(
            "물리에서 일은 무엇을 의미해?",
            "역학적 에너지 보존은 어떤 조건에서 성립해?",
            "마찰이 있으면 역학적 에너지는 어떻게 돼?",
        ),
    ),
    LectureSample(
        slug="physics-week3-electricity",
        course_title="물리학",
        session_title="물리학 3주차 - 전기장과 회로",
        recording_title="물리학 3주차 전기장과 회로 녹음본",
        chunks=(
            "물리학 3주차에서는 전기장과 회로의 기초를 다룹니다. 전하 사이에는 전기력이 작용하며, 같은 종류의 전하는 서로 밀어내고 다른 종류의 전하는 서로 끌어당깁니다. 전기장은 어떤 위치에 양의 시험 전하를 놓았을 때 그 전하가 받는 힘을 단위 전하당 힘으로 나타낸 물리량입니다.",
            "전위는 단위 전하가 가지는 전기적 위치에너지입니다. 전위차가 존재하면 전하가 이동할 수 있는 원인이 생기고, 회로에서는 이 전위차를 전압이라고 부릅니다. 전류는 단위 시간 동안 도선을 지나는 전하량이며, 관습적으로 양전하가 이동하는 방향을 전류의 방향으로 정의합니다.",
            "옴의 법칙은 전압이 전류와 저항의 곱과 같다는 관계입니다. 같은 저항에서 전압이 커지면 전류도 비례해서 증가합니다. 직렬 회로에서는 전류가 모든 저항에 동일하게 흐르고, 병렬 회로에서는 각 가지의 전압이 동일하게 걸립니다.",
            "전력은 단위 시간당 소비되는 에너지이며 전압과 전류의 곱으로 계산할 수 있습니다. 전기 기기의 소비 전력이 클수록 같은 시간 동안 더 많은 전기에너지를 사용합니다. 다음 실습에서는 직렬 회로와 병렬 회로를 구성하고 각 저항에 걸리는 전압과 전류를 측정합니다.",
        ),
        questions=(
            "전기장은 무엇을 의미해?",
            "전압과 전류는 각각 어떻게 설명했어?",
            "직렬 회로와 병렬 회로의 차이는 뭐야?",
        ),
    ),
)


def deterministic_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{SAMPLE_PREFIX}:{name}")


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    remain = total % 60
    return f"{minutes}:{remain:02d}"


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remain = total % 60
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def week_label(value: date) -> str:
    return f"{value.year}. {value.month}. {value.day}. {WEEKDAYS[value.weekday()]}"


def ensure_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS courses (
            course_id UUID PRIMARY KEY,
            user_id UUID NULL,
            parent_course_id UUID NULL,
            title TEXT NOT NULL,
            type TEXT NULL,
            description TEXT NULL,
            color TEXT NULL,
            icon TEXT NULL,
            created_at TIMESTAMP NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id UUID PRIMARY KEY,
            course_id UUID NULL,
            session_date DATE NULL,
            title TEXT NULL,
            duration_sec INTEGER NULL DEFAULT 0,
            status TEXT NULL,
            created_at TIMESTAMP NULL DEFAULT NOW(),
            file_kind VARCHAR(50) NULL,
            tag VARCHAR(50) NULL,
            icon VARCHAR(50) NULL,
            color VARCHAR(50) NULL,
            session_pdf JSONB NULL DEFAULT '[]'::jsonb,
            session_voicefile JSONB NULL DEFAULT '[]'::jsonb,
            summary_notes JSONB NULL DEFAULT '[]'::jsonb
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            transcript_id UUID PRIMARY KEY,
            session_id UUID NULL,
            recording_id TEXT NULL,
            chunk_index INTEGER NULL,
            start_time DOUBLE PRECISION NULL,
            end_time DOUBLE PRECISION NULL,
            original_text TEXT NULL,
            chunk_text TEXT NULL,
            corrected_text TEXT NULL,
            confidence DOUBLE PRECISION NULL,
            speaker_id TEXT NULL,
            speaker_name TEXT NULL,
            created_at TIMESTAMP NULL DEFAULT NOW()
        )
        """
    )
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS user_id UUID NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS parent_course_id UUID NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS type TEXT NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS description TEXT NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    cur.execute("ALTER TABLE courses ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS course_id UUID NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_date DATE NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duration_sec INTEGER NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status TEXT NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS file_kind VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tag VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS icon VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS color VARCHAR(50) NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_pdf JSONB NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_voicefile JSONB NULL")
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_notes JSONB NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS recording_id TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_index INTEGER NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS start_time DOUBLE PRECISION NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS end_time DOUBLE PRECISION NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS original_text TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS chunk_text TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS corrected_text TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_id TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_name TEXT NULL")
    cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NULL")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_course_id ON sessions(course_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_id ON transcripts(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_recording_id ON transcripts(recording_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session_chunk ON transcripts(session_id, chunk_index)")


def remove_legacy_root_folder(cur) -> None:
    """이전 시드의 BridgePRAG 루트 폴더를 제거하고 과목 폴더를 최상위로 올립니다."""
    legacy_id = deterministic_uuid(f"course:{FOLDER_TITLE}")
    cur.execute(
        "UPDATE courses SET parent_course_id = NULL WHERE parent_course_id = %s",
        (legacy_id,),
    )
    cur.execute(
        "DELETE FROM courses WHERE course_id = %s AND title = %s",
        (legacy_id, FOLDER_TITLE),
    )


def upsert_course(cur, title: str, *, parent_id: uuid.UUID | None = None) -> uuid.UUID:
    course_id = deterministic_uuid(f"course:{title}")
    cur.execute(
        """
        INSERT INTO courses (
            course_id, user_id, parent_course_id, title, type, description, color, icon, created_at
        )
        VALUES (%s, NULL, %s, %s, %s, %s, '#2563eb', %s, %s)
        ON CONFLICT (course_id) DO UPDATE
        SET parent_course_id = EXCLUDED.parent_course_id,
            title = EXCLUDED.title,
            type = EXCLUDED.type,
            description = EXCLUDED.description,
            color = EXCLUDED.color,
            icon = EXCLUDED.icon
        """,
        (
            course_id,
            parent_id,
            title,
            "folder",
            f"{title} 샘플 강의",
            "folder" if parent_id is None else "school",
            datetime.now(),
        ),
    )
    return course_id


def build_rows(sample: LectureSample, session_id: uuid.UUID, recording_id: str, started_at: datetime):
    transcriptions: list[dict] = []
    db_rows: list[tuple] = []
    cursor = 0.0

    for index, text in enumerate(sample.chunks):
        duration = max(18.0, min(70.0, len(text) / 7.0))
        start_time = round(cursor, 2)
        end_time = round(cursor + duration, 2)
        cursor = end_time + 2.0
        transcript_id = deterministic_uuid(f"transcript:{sample.slug}:{index}")
        created_at = started_at + timedelta(seconds=start_time)
        speaker_id = "professor"
        speaker_name = "교수자"

        transcriptions.append({
            "id": f"{recording_id}-{index}",
            "recordingId": recording_id,
            "time": format_time(start_time),
            "speakerId": speaker_id,
            "speaker": speaker_name,
            "text": text,
            "segments": [{
                "id": str(transcript_id),
                "text": text,
                "status": "confirmed",
                "transcript_id": str(transcript_id),
                "transcriptId": str(transcript_id),
                "chunk_index": index,
                "start": start_time,
                "end": end_time,
                "start_time": start_time,
                "end_time": end_time,
            }],
        })
        db_rows.append((
            transcript_id,
            session_id,
            recording_id,
            index,
            start_time,
            end_time,
            speaker_id,
            speaker_name,
            text,
            text,
            created_at,
        ))

    return transcriptions, db_rows, int(round(max(cursor - 2.0, 0)))


def session_voicefile(sample: LectureSample, recording_id: str, transcriptions: list[dict], duration_sec: int):
    today = date.today()
    now = datetime.now().replace(microsecond=0)
    week_id = f"week-{today.isoformat()}"
    return [{
        "weekId": week_id,
        "id": week_id,
        "weekKey": today.isoformat(),
        "label": "샘플",
        "dateLabel": week_label(today),
        "expanded": True,
        "materialFolderExpanded": True,
        "recordingFolderExpanded": True,
        "recordings": [{
            "id": recording_id,
            "recordingId": recording_id,
            "title": sample.recording_title,
            "startedAt": now.isoformat(),
            "endedAt": (now + timedelta(seconds=duration_sec)).isoformat(),
            "durationText": format_duration(duration_sec),
            "recordingMode": "lecture",
            "diarizationEnabled": False,
            "materialIds": [],
            "materialNames": [],
            "audioUrl": None,
            "transcriptions": transcriptions,
        }],
    }]


def write_transcript_jsonl(session_id: uuid.UUID, rows: list[tuple]) -> None:
    output_dir = BACKEND_ROOT / "data" / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}.jsonl"
    lines = []
    for row in rows:
        lines.append({
            "transcript_id": str(row[0]),
            "session_id": str(row[1]),
            "recording_id": row[2],
            "start_time": row[4],
            "end_time": row[5],
            "speaker_id": row[6],
            "speaker_name": row[7],
            "raw_text": row[8],
            "text": row[9],
            "created_at": row[10].isoformat(),
        })
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")


def index_transcripts(rows: list[tuple], *, course_title: str, session_title: str, session_date: date) -> int:
    """Insert seeded chunks into the same vector store used by service RAG."""
    from rag_search import add_document

    indexed = 0
    for row in rows:
        (
            transcript_id,
            session_id,
            recording_id,
            chunk_index,
            start_time,
            end_time,
            _speaker_id,
            _speaker_name,
            _raw_text,
            text,
            created_at,
        ) = row
        add_document(
            text,
            {
                "source_type": "transcript",
                "transcript_id": str(transcript_id),
                "session_id": str(session_id),
                "recording_id": str(recording_id),
                "course_title": course_title,
                "session_title": session_title,
                "session_date": str(session_date),
                "chunk_index": int(chunk_index),
                "start_time": float(start_time),
                "end_time": float(end_time),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
            },
        )
        indexed += 1
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed BridgePRAG lecture-like workspace samples.")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only write PostgreSQL rows. Skip BGE/PGVector indexing.",
    )
    args = parser.parse_args()

    inserted = []
    pending_index_rows: list[tuple[LectureSample, list[tuple]]] = []
    config = psycopg2_config()
    with psycopg2.connect(**config) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur)
            remove_legacy_root_folder(cur)
            for sample in SAMPLES:
                course_id = upsert_course(cur, sample.course_title)
                session_id = deterministic_uuid(f"session:{sample.slug}")
                recording_id = f"{SAMPLE_PREFIX}-{sample.slug}"
                started_at = datetime.now().replace(microsecond=0)
                transcriptions, transcript_rows, duration_sec = build_rows(sample, session_id, recording_id, started_at)

                cur.execute(
                    """
                    INSERT INTO sessions (
                        session_id, course_id, session_date, title, duration_sec, status, created_at,
                        file_kind, tag, icon, color, session_pdf, session_voicefile, summary_notes
                    )
                    VALUES (%s, %s, %s, %s, %s, 'created', %s,
                            'lecture', '수업', 'article', '#2563eb', '[]'::jsonb, %s::jsonb, '[]'::jsonb)
                    ON CONFLICT (session_id) DO UPDATE
                    SET course_id = EXCLUDED.course_id,
                        session_date = EXCLUDED.session_date,
                        title = EXCLUDED.title,
                        duration_sec = EXCLUDED.duration_sec,
                        status = EXCLUDED.status,
                        file_kind = EXCLUDED.file_kind,
                        tag = EXCLUDED.tag,
                        icon = EXCLUDED.icon,
                        color = EXCLUDED.color,
                        session_voicefile = EXCLUDED.session_voicefile
                    """,
                    (
                        session_id,
                        course_id,
                        date.today(),
                        sample.session_title,
                        duration_sec,
                        datetime.now(),
                        Json(session_voicefile(sample, recording_id, transcriptions, duration_sec)),
                    ),
                )
                cur.execute(
                    "DELETE FROM transcripts WHERE session_id = %s AND recording_id = %s",
                    (session_id, recording_id),
                )
                execute_batch(
                    cur,
                    """
                    INSERT INTO transcripts (
                        transcript_id, session_id, recording_id, chunk_index, start_time, end_time,
                        speaker_id, speaker_name, chunk_text, corrected_text, created_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (transcript_id) DO UPDATE
                    SET recording_id = EXCLUDED.recording_id,
                        chunk_index = EXCLUDED.chunk_index,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        speaker_id = EXCLUDED.speaker_id,
                        speaker_name = EXCLUDED.speaker_name,
                        chunk_text = EXCLUDED.chunk_text,
                        corrected_text = EXCLUDED.corrected_text,
                        created_at = EXCLUDED.created_at
                    """,
                    transcript_rows,
                    page_size=50,
                )
                write_transcript_jsonl(session_id, transcript_rows)
                pending_index_rows.append((sample, transcript_rows))
                inserted.append({
                    "course": sample.course_title,
                    "session": sample.session_title,
                    "session_id": str(session_id),
                    "recording": sample.recording_title,
                    "recording_id": recording_id,
                    "chunks": len(sample.chunks),
                    "sample_questions": list(sample.questions),
                })
        conn.commit()

    indexed_chunks = 0
    if not args.skip_index:
        for sample, rows in pending_index_rows:
            indexed_chunks += index_transcripts(
                rows,
                course_title=sample.course_title,
                session_title=sample.session_title,
                session_date=date.today(),
            )

    print(json.dumps({
        "ok": True,
        "database": config.get("database"),
        "folders": sorted({sample.course_title for sample in SAMPLES}),
        "indexed_chunks": indexed_chunks,
        "index_skipped": bool(args.skip_index),
        "inserted": inserted,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
