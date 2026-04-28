import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_api.workspace.router import router as workspace_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    # testmain.py가 정상 실행 중인지 확인하는 테스트 엔드포인트입니다.
    return {"ok": True, "service": "workspace-fastapi-test"}


# 나중에 main.py로 옮길 때도 아래 두 줄 구조만 동일하게 사용하면 됩니다.
app.include_router(workspace_router)
