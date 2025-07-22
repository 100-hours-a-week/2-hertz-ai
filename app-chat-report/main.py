# -*- coding: utf-8 -*-
"""
애플리케이션 초기화 및 구성 / API 서버 시작점 역할 수행
애플리케이션 인스턴스 생성, 라우터 등록, 환경 설정 로드 등 담당
"""


import os
from contextlib import asynccontextmanager

from api.endpoints.chat_report_router import router as chat_report_router
from db.mongodb import mongodb  # mongodb 인스턴스 임포트
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from utils.error_handler import register_exception_handlers

# .env 파일에서 환경 변수 로드
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    # (db/mongodb.py에서 이미 연결은 초기화됨)
    yield
    # 앱 종료 시 실행
    mongodb.close()


# --- FastAPI 애플리케이션 설정 ---

# FastAPI 앱 인스턴스 생성 (Swagger UI 문서에 표시될 메타데이터 포함)
app = FastAPI(
    lifespan=lifespan,
    title="TUNING Chat Report API",
    description="조직 내부 사용자 간의 자연스럽고 부담 없는 소통을 돕는 소셜 매칭 서비스 API",
    version="1.0.0",
)

register_exception_handlers(app)

# 라우터 등록
app.include_router(chat_report_router)


# 루트 경로 핸들러 - 개발 환경에서는 API 문서(Swagger)로 리다이렉트, 프로덕션에서는 접근 제한
@app.get("/")
async def root():
    env = os.getenv("ENVIRONMENT")
    if env == "dev":
        return RedirectResponse(url="/docs")
    else:
        return HTTPException(status_code=404)
