# -*- coding: utf-8 -*-
"""
애플리케이션 초기화 및 구성 / API 서버 시작점 역할 수행
애플리케이션 인스턴스 생성, 라우터 등록, 환경 설정 로드 등 담당
"""


import os

from api.endpoints.health_router import HealthRouter
from api.endpoints.monitoring_router import PerformanceRouter
from api.endpoints.tuning_router import TuningRouter
from api.endpoints.user_router import UserRouter
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from utils.error_handler import register_exception_handlers

# .env 파일에서 환경 변수 로드
load_dotenv()

# FastAPI 앱 인스턴스 생성 (Swagger UI 문서에 표시될 메타데이터 포함)
app = FastAPI(
    title="TUNING API",
    description="조직 내부 사용자 간의 자연스럽고 부담 없는 소통을 돕는 소셜 매칭 서비스 API",
    version="1.0.0",
)

register_exception_handlers(app)  # 반드시 포함

# 라우터 등록 - API를 기능별로 모듈화
app.include_router(HealthRouter().router)
app.include_router(UserRouter().router)
app.include_router(TuningRouter().router)
app.include_router(PerformanceRouter().router)


# 루트 경로 핸들러 - 개발 환경에서는 API 문서(Swagger)로 리다이렉트, 프로덕션에서는 접근 제한
@app.get("/")
async def root():
    env = os.getenv("ENVIRONMENT")
    if env == "dev":
        return RedirectResponse(url="/docs")
    else:
        return HTTPException(status_code=404)
