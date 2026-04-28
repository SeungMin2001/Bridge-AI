import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).with_name(".env"))

mcp = FastMCP("LectoAI_Notion_Schedule")

NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_DATABASE_API_URL = "https://api.notion.com/v1/databases"
NOTION_DATA_SOURCE_API_URL = "https://api.notion.com/v1/data_sources"
NOTION_VERSION = "2025-09-03"


def _get_env(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(f"{name} is not set")
  return value


def _normalize_end_time(start: str, end: str | None) -> str:
  if end:
    return end

  start_at = datetime.fromisoformat(start)
  return (start_at + timedelta(hours=1)).isoformat()


def _notion_headers(token: str) -> dict:
  return {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION,
  }


def _get_database_properties(token: str, database_id: str) -> dict:
  response = requests.get(
    f"{NOTION_DATABASE_API_URL}/{database_id}",
    headers=_notion_headers(token),
    timeout=15,
  )
  if not response.ok:
    raise RuntimeError(
      f"Notion database error {response.status_code}: {response.text}"
    )
  data = response.json()
  data_sources = data.get("data_sources", [])
  if not data_sources:
    raise RuntimeError("No data sources were found in this Notion database.")

  return _get_data_source_properties(token, data_sources[0]["id"])


def _get_data_source_properties(token: str, data_source_id: str) -> dict:
  response = requests.get(
    f"{NOTION_DATA_SOURCE_API_URL}/{data_source_id}",
    headers=_notion_headers(token),
    timeout=15,
  )
  if not response.ok:
    raise RuntimeError(
      f"Notion data source error {response.status_code}: {response.text}"
    )

  return response.json().get("properties", {})


def _resolve_data_source_id(token: str, database_id: str) -> str:
  response = requests.get(
    f"{NOTION_DATABASE_API_URL}/{database_id}",
    headers=_notion_headers(token),
    timeout=15,
  )
  if not response.ok:
    raise RuntimeError(
      f"Notion database error {response.status_code}: {response.text}"
    )

  data_sources = response.json().get("data_sources", [])
  if not data_sources:
    raise RuntimeError("No data sources were found in this Notion database.")

  return data_sources[0]["id"]


def _build_date_property(property_type: str, start: str, end: str | None) -> dict:
  if property_type == "date":
    return {
      "date": {
        "start": start,
        "end": _normalize_end_time(start, end),
      },
    }

  if property_type == "rich_text":
    return {
      "rich_text": [
        {
          "text": {
            "content": f"{start} - {_normalize_end_time(start, end)}",
          },
        }
      ],
    }

  raise RuntimeError(
    f"Property '{os.getenv('NOTION_DATE_PROPERTY', 'Date')}' must be a Notion date property. "
    f"Current type is '{property_type}'."
  )


def _create_notion_schedule(title: str, start: str, end: str | None = None) -> dict:
  token = _get_env("NOTION_TOKEN")
  database_id = _get_env("NOTION_DATABASE_ID")
  title_property = os.getenv("NOTION_TITLE_PROPERTY", "Name")
  date_property = os.getenv("NOTION_DATE_PROPERTY", "Date")
  data_source_id = os.getenv("NOTION_DATA_SOURCE_ID") or _resolve_data_source_id(token, database_id)
  properties = _get_data_source_properties(token, data_source_id)
  date_schema = properties.get(date_property)

  if not date_schema:
    raise RuntimeError(
      f"Property '{date_property}' was not found. Available properties: {', '.join(properties.keys())}"
    )

  response = requests.post(
    NOTION_API_URL,
    headers=_notion_headers(token),
    json={
      "parent": {
        "type": "data_source_id",
        "data_source_id": data_source_id,
      },
      "properties": {
        title_property: {
          "title": [
            {
              "text": {
                "content": title,
              },
            }
          ],
        },
        date_property: _build_date_property(date_schema.get("type"), start, end),
      },
    },
    timeout=15,
  )
  if not response.ok:
    raise RuntimeError(
      f"Notion API error {response.status_code}: {response.text}"
    )
  data = response.json()

  return {
    "ok": True,
    "notion_page_id": data.get("id"),
    "url": data.get("url"),
    "title": title,
    "start": start,
    "end": _normalize_end_time(start, end),
  }


def create_notion_schedule(title: str, start: str, end: str | None = None) -> dict:
  return _create_notion_schedule(title=title, start=start, end=end)


@mcp.tool()
def create_test_schedule(title: str, start: str, end: str | None = None) -> dict:
  """
  Creates one simple schedule item in a Notion database.

  start/end must be ISO-8601 strings, for example:
  2026-04-29T15:00:00+09:00
  """
  return _create_notion_schedule(title=title, start=start, end=end)


@mcp.tool()
def create_schedule_from_sentence(sentence: str, start: str) -> dict:
  """
  Sends a simple sentence to Notion as a schedule title.

  This is only for an integration smoke test, not AI extraction.
  """
  return _create_notion_schedule(title=sentence, start=start)


def run_smoke_test() -> None:
  result = _create_notion_schedule(
    title="수학일정관리",
    start="2026-04-29T07:00:00+09:00",
    end="2026-04-29T10:00:00+09:00",
  )
  print(result)


if __name__ == "__main__":
  if "--test" in sys.argv:
    run_smoke_test()
  else:
    mcp.run(transport="stdio")
