import asyncio
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_api.workspace.common import WorkspaceApiError
from db_api.workspace.courses_api import create_course, delete_course, update_course
from db_api.workspace.sessions_api import create_session_file, delete_session_file, update_session_resources
from db_api.workspace.tree_api import get_workspace_tree


HOST = os.getenv("WORKSPACE_API_HOST", "127.0.0.1")
PORT = int(os.getenv("WORKSPACE_API_PORT", "8001"))


API_LOOP = asyncio.new_event_loop()


def _run_api_loop() -> None:
    asyncio.set_event_loop(API_LOOP)
    API_LOOP.run_forever()


API_LOOP_THREAD = threading.Thread(target=_run_api_loop, name="workspace-api-loop", daemon=True)
API_LOOP_THREAD.start()


class WorkspaceRequestHandler(BaseHTTPRequestHandler):
    # main.py를 띄우지 않고 워크스페이스 DB 저장만 확인하는 테스트 HTTP 서버입니다.
    def _send_json(self, status_code: int, payload: dict) -> None:
        # 모든 응답을 JSON으로 내려주고, 프론트 개발 서버에서 접근할 수 있게 CORS를 열어둡니다.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        # POST body를 JSON으로 읽습니다. body가 없으면 빈 dict로 처리합니다.
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}

        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise WorkspaceApiError("Request body must be valid JSON.") from exc

    def _run_api(self, coroutine):
        # get_pool()이 같은 이벤트 루프에서 재사용되도록 모든 async DB 작업을 전용 루프에서 실행합니다.
        future = asyncio.run_coroutine_threadsafe(coroutine, API_LOOP)
        return future.result()

    def _handle_error(self, error: Exception) -> None:
        # 검증 오류는 지정된 status code로, 그 외 예외는 500으로 내려줍니다.
        if isinstance(error, WorkspaceApiError):
            print(f"[workspace-api] handled error: {error}")
            self._send_json(error.status_code, {"ok": False, "error": str(error)})
            return

        print("[workspace-api] unexpected error:")
        traceback.print_exc()
        self._send_json(500, {"ok": False, "error": str(error)})

    def do_OPTIONS(self) -> None:
        # 브라우저 preflight 요청 처리용입니다.
        self._send_json(204, {})

    def do_GET(self) -> None:
        # GET /health, GET /workspace/tree를 처리합니다.
        path = urlparse(self.path).path

        try:
            if path == "/health":
                self._send_json(200, {"ok": True, "service": "workspace"})
                return

            if path == "/workspace/tree":
                result = self._run_api(get_workspace_tree())
                self._send_json(200, result)
                return

            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as error:
            self._handle_error(error)

    def do_POST(self) -> None:
        # POST /workspace/courses, POST /workspace/sessions를 처리합니다.
        path = urlparse(self.path).path

        try:
            payload = self._read_json()

            if path == "/workspace/courses":
                result = self._run_api(create_course(payload))
                self._send_json(201, result)
                return

            if path == "/workspace/sessions":
                result = self._run_api(create_session_file(payload))
                self._send_json(201, result)
                return

            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as error:
            self._handle_error(error)

    def do_PUT(self) -> None:
        # PUT /workspace/courses/{course_id}, PUT /workspace/sessions/{session_id}/resources 처리
        path = urlparse(self.path).path

        try:
            payload = self._read_json()

            if path.startswith("/workspace/sessions/") and path.endswith("/resources"):
                session_id = path.split("/")[-2]
                result = self._run_api(update_session_resources(session_id, payload))
                self._send_json(200, result)
                return

            if path.startswith("/workspace/courses/"):
                course_id = path.rsplit("/", 1)[-1]
                result = self._run_api(update_course(course_id, payload))
                self._send_json(200, result)
                return

            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as error:
            self._handle_error(error)

    def do_DELETE(self) -> None:
        # DELETE /workspace/courses/{course_id}, DELETE /workspace/sessions/{session_id}를 처리합니다.
        path = urlparse(self.path).path

        try:
            if path.startswith("/workspace/courses/"):
                course_id = path.rsplit("/", 1)[-1]
                result = self._run_api(delete_course(course_id))
                self._send_json(200, result)
                return

            if path.startswith("/workspace/sessions/"):
                session_id = path.rsplit("/", 1)[-1]
                result = self._run_api(delete_session_file(session_id))
                self._send_json(200, result)
                return

            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as error:
            self._handle_error(error)

    def log_message(self, format, *args) -> None:
        print(f"[workspace-api] {self.address_string()} - {format % args}")


def run() -> None:
    # DB 연동 테스트 서버를 127.0.0.1:8001에서 실행합니다.
    server = ThreadingHTTPServer((HOST, PORT), WorkspaceRequestHandler)
    print(f"Workspace test server running at http://{HOST}:{PORT}")
    print("Endpoints: GET /health, GET /workspace/tree, POST /workspace/courses, PUT /workspace/courses/{id}, DELETE /workspace/courses/{id}, POST /workspace/sessions, PUT /workspace/sessions/{id}/resources, DELETE /workspace/sessions/{id}")
    server.serve_forever()


if __name__ == "__main__":
    run()
