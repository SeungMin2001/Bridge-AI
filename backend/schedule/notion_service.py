"""
노션 API 연동 서비스 (노션 캘린더)

환경변수:
  - NOTION_API_KEY      : 노션 통합 API 토큰
  - NOTION_DATABASE_ID  : 등록 대상 노션 데이터베이스 ID
"""
import os
import httpx
import logging
import string
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env", override=False)

NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
NOTION_VERSION = "2022-06-28"

# 노션 데이터베이스 속성(컬럼) 이름 환경변수 (기본값은 한국어 노션 기본값 설정)
PROP_TITLE = os.getenv("NOTION_PROP_TITLE", "이름")
PROP_DATE = os.getenv("NOTION_PROP_DATE", "날짜")
PROP_DESC = os.getenv("NOTION_PROP_DESC", "설명")
PROP_TYPE = os.getenv("NOTION_PROP_TYPE", "구분")


def normalize_notion_id(value: str | None) -> str:
    """Accept pasted Notion IDs/URLs and normalize the database/page ID part."""
    notion_id = str(value or "").strip()
    if not notion_id:
        return ""

    notion_id = notion_id.split("?")[0].rstrip("/").split("/")[-1]
    allowed = set(string.hexdigits + "-")
    if notion_id.startswith("v") and len(notion_id) > 1 and all(ch in allowed for ch in notion_id[1:]):
        notion_id = notion_id[1:]
    return notion_id


def get_notion_settings() -> tuple[str, str]:
    api_key = os.getenv("NOTION_API_KEY", NOTION_API_KEY).strip()
    db_id = normalize_notion_id(os.getenv("NOTION_DATABASE_ID", NOTION_DATABASE_ID))
    return api_key, db_id


async def resolve_database_properties(client: httpx.AsyncClient, headers: dict, db_id: str) -> dict:
    """Resolve actual Notion property names by type, while honoring env overrides."""
    defaults = {
        "title": os.getenv("NOTION_PROP_TITLE", PROP_TITLE),
        "date": os.getenv("NOTION_PROP_DATE", PROP_DATE),
        "desc": os.getenv("NOTION_PROP_DESC", PROP_DESC),
        "type": os.getenv("NOTION_PROP_TYPE", PROP_TYPE),
    }

    try:
        res = await client.get(f"https://api.notion.com/v1/databases/{db_id}", headers=headers)
        if res.status_code >= 300:
            logger.warning("[NOTION] 데이터베이스 속성 조회 실패 (%s): %s", res.status_code, res.text)
            return defaults
        properties = res.json().get("properties", {})
    except Exception as exc:
        logger.warning("[NOTION] 데이터베이스 속성 조회 중 오류: %s", exc)
        return defaults

    def pick(env_name: str, notion_type: str, required: bool = False) -> str | None:
        if env_name in properties and properties[env_name].get("type") == notion_type:
            return env_name
        for name, schema in properties.items():
            if schema.get("type") == notion_type:
                return name
        return env_name if required else None

    return {
        "title": pick(defaults["title"], "title", required=True),
        "date": pick(defaults["date"], "date"),
        "desc": pick(defaults["desc"], "rich_text"),
        "type": pick(defaults["type"], "select"),
    }


async def sync_schedule_to_notion(schedule: dict) -> str:
    """
    일정 데이터를 노션 데이터베이스에 등록하고 생성된 페이지 ID를 반환합니다.
    """
    api_key, db_id = get_notion_settings()

    if not api_key:
        raise ValueError("NOTION_API_KEY 환경변수가 설정되지 않았습니다.")
    if not db_id:
        raise ValueError("NOTION_DATABASE_ID 환경변수가 설정되지 않았습니다.")

    url = "https://api.notion.com/v1/pages"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    # 날짜 포맷팅 (ISO 포맷 혹은 YYYY-MM-DD)
    due_date_str = schedule.get("due_date")
    notion_date = None
    if due_date_str:
        try:
            dt = datetime.fromisoformat(due_date_str)
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                notion_date = {"start": dt.strftime("%Y-%m-%d")}
            else:
                notion_date = {"start": dt.isoformat()}
        except Exception as e:
            logger.warning(f"[NOTION] 날짜 포맷 변환 실패: {e}")
            # 파싱 실패 시 단순 날짜 부분만 분리해서 시도
            notion_date = {"start": due_date_str.split("T")[0]}

    async with httpx.AsyncClient(timeout=15.0) as client:
        notion_props = await resolve_database_properties(client, headers, db_id)

        # 노션 데이터베이스 속성 구성
        properties = {
            notion_props["title"]: {
                "title": [
                    {
                        "text": {
                            "content": schedule.get("title", "제목 없음")
                        }
                    }
                ]
            }
        }

        if notion_date and notion_props.get("date"):
            properties[notion_props["date"]] = {
                "date": notion_date
            }

        description = schedule.get("description")
        if description and notion_props.get("desc"):
            properties[notion_props["desc"]] = {
                "rich_text": [
                    {
                        "text": {
                            "content": description
                        }
                    }
                ]
            }

        event_type = schedule.get("event_type")
        if event_type and notion_props.get("type"):
            properties[notion_props["type"]] = {
                "select": {
                    "name": event_type
                }
            }

        payload = {
            "parent": {"database_id": db_id},
            "properties": properties,
        }

        # 본문 블록에 전사문 출처 내용 추가 (상세 분석용)
        source_text = schedule.get("source_text")
        if source_text:
            payload["children"] = [
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"text": {"content": "출처 문장 (Context)"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": source_text}}]
                    }
                }
            ]

        logger.info(f"[NOTION] 페이지 등록 요청 전송: db_id={db_id}, title={schedule.get('title')}")

        res = await client.post(url, headers=headers, json=payload)
        if res.status_code >= 300:
            logger.error(f"[NOTION] API 호출 실패 ({res.status_code}): {res.text}")
            try:
                err_msg = res.json().get("message", res.text)
            except Exception:
                err_msg = res.text
            raise RuntimeError(f"노션 등록 실패: {err_msg}")

        data = res.json()
        page_id = data.get("id")
        logger.info(f"[NOTION] 페이지 생성 성공: page_id={page_id}")
        return page_id


async def fetch_schedules_from_notion() -> list[dict]:
    """
    노션 캘린더 데이터베이스의 모든 일정을 조회하여
    우리 스케줄 형식의 딕셔너리 리스트로 반환한다.

    각 항목에는 notion_page_id가 포함되어 있어
    DB의 기존 일정과 비교하여 중복을 필터링할 수 있다.
    """
    api_key, db_id = get_notion_settings()

    if not api_key:
        raise ValueError("NOTION_API_KEY 환경변수가 설정되지 않았습니다.")
    if not db_id:
        raise ValueError("NOTION_DATABASE_ID 환경변수가 설정되지 않았습니다.")

    url = f"https://api.notion.com/v1/databases/{db_id}/query"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    all_pages = []
    has_more = True
    start_cursor = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        notion_props = await resolve_database_properties(client, headers, db_id)
        while has_more:
            body = {}
            if start_cursor:
                body["start_cursor"] = start_cursor

            res = await client.post(url, headers=headers, json=body)
            if res.status_code >= 300:
                logger.error(f"[NOTION] 캘린더 조회 실패 ({res.status_code}): {res.text}")
                try:
                    err_msg = res.json().get("message", res.text)
                except Exception:
                    err_msg = res.text
                raise RuntimeError(f"노션 캘린더 조회 실패: {err_msg}")

            data = res.json()
            all_pages.extend(data.get("results", []))
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

    logger.info(f"[NOTION] 캘린더에서 {len(all_pages)}개 일정 조회 완료")

    # 노션 페이지를 우리 스케줄 형식으로 변환
    schedules = []
    for page in all_pages:
        try:
            props = page.get("properties", {})
            schedule = _parse_notion_page(
                page["id"],
                props,
                notion_props["title"],
                notion_props.get("date"),
                notion_props.get("desc"),
                notion_props.get("type"),
            )
            schedules.append(schedule)
        except Exception as e:
            logger.warning(f"[NOTION] 페이지 파싱 실패 (id={page.get('id')}): {e}")
            continue

    return schedules


def _parse_notion_page(
    page_id: str,
    props: dict,
    prop_title: str,
    prop_date: str,
    prop_desc: str,
    prop_type: str,
) -> dict:
    """노션 페이지 properties를 우리 스케줄 형식 딕셔너리로 변환한다."""
    # 제목 파싱
    title = ""
    title_prop = props.get(prop_title, {})
    if title_prop.get("type") == "title":
        title_parts = title_prop.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_parts)

    # 날짜 파싱
    due_date = None
    date_prop = props.get(prop_date, {})
    if date_prop.get("type") == "date" and date_prop.get("date"):
        due_date = date_prop["date"].get("start")

    # 설명 파싱
    description = None
    desc_prop = props.get(prop_desc, {})
    if desc_prop.get("type") == "rich_text":
        desc_parts = desc_prop.get("rich_text", [])
        if desc_parts:
            description = "".join(t.get("plain_text", "") for t in desc_parts)

    # 구분(타입) 파싱
    event_type = None
    type_prop = props.get(prop_type, {})
    if type_prop.get("type") == "select" and type_prop.get("select"):
        event_type = type_prop["select"].get("name")

    return {
        "notion_page_id": page_id,
        "title": title or "제목 없음",
        "due_date": due_date,
        "description": description,
        "event_type": event_type,
    }
