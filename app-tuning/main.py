# -*- coding: utf-8 -*-
"""
애플리케이션 초기화 및 구성 / API 서버 시작점 역할 수행
애플리케이션 인스턴스 생성, 라우터 등록, 환경 설정 로드 등 담당
"""


import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

from api.endpoints.health_router import HealthRouter
from api.endpoints.monitoring_router import PerformanceRouter
from api.endpoints.tuning_router import TuningRouter
from api.endpoints.user_router import UserRouter
from core.vector_database.client import get_chroma_client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from scripts.recompute_all_similarities_optimized import (
    recompute_all_similarities_optimized_v2,
)
from utils.error_handler import register_exception_handlers
from utils.logger import logger, logging

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로거 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 디버깅용 전역 플래그
LIFESPAN_CALLED = False
STARTUP_EVENT_CALLED = False


async def monitor_script_execution(process):
    """스크립트 실행 상태 모니터링"""
    try:
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info("✅ 유사도 재계산 스크립트 실행 완료")
            if stdout:
                logger.info(f"스크립트 출력: {stdout.decode()}")
        else:
            logger.error(
                f"❌ 유사도 재계산 스크립트 실행 실패 (exit code: {process.returncode})"
            )
            if stderr:
                logger.error(f"에러 출력: {stderr.decode()}")
    except Exception as e:
        logger.error(f"스크립트 모니터링 중 오류: {e}")


async def run_similarity_script():
    """유사도 재계산 스크립트 실행"""

    try:
        script_path = "scripts/recompute_all_similarities_optimized.py"

        # 스크립트 파일 존재 확인
        if not os.path.exists(script_path):
            logger.warning(f"경고: {script_path} 파일을 찾을 수 없습니다.")
            return False

        logger.info("유사도 재계산 스크립트 실행 시작...")
        # 시작 시간 기록
        start_time = time.time()

        # 환경 변수 전달
        env = os.environ.copy()

        # 비동기 프로세스 실행
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        # 스크립트가 완료될 때까지 직접 기다리고, 표준 출력/에러를 가져옵니다.
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info("✅ 유사도 재계산 스크립트 실행 완료")
            elapsed = round(time.time() - start_time, 3)
            logger.info(f"스크립트 실행 시간: {elapsed}초")
        else:
            logger.error(
                f"❌ 유사도 재계산 스크립트 실행 실패 (exit code: {process.returncode})"
            )
            if stderr:
                logger.error(f"에러 출력:\n{stderr.decode()}")
            return False

    except Exception as e:
        logger.error(f"유사도 재계산 스크립트 실행 중 오류 발생: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global LIFESPAN_CALLED
    LIFESPAN_CALLED = True

    # 시작 시 실행
    logger.info("🚀 [LIFESPAN] 애플리케이션 시작 중...")
    print("🚀 [LIFESPAN] 애플리케이션 시작 중...")

    # 환경 변수 체크
    env = os.getenv("ENVIRONMENT", "dev")
    logger.info(f"[LIFESPAN] Environment: {env}")

    try:
        logger.info("[LIFESPAN] ChromaDB 연결 확인 시작...")
        if get_chroma_client():
            logger.info("[LIFESPAN] ChromaDB 연결 성공, 스크립트 실행...")
            recompute_all_similarities_optimized_v2()
        else:
            logger.warning(
                "[LIFESPAN] ⚠️ ChromaDB 연결 실패로 인해 유사도 재계산 스크립트를 실행하지 않습니다."
            )
    except Exception as e:
        logger.error(f"[LIFESPAN] startup 중 오류: {e}")
        import traceback

        # 경과 시간 계산

        logger.error(f"[LIFESPAN] 상세 오류: {traceback.format_exc()}")

    logger.info("✅ [LIFESPAN] 애플리케이션 시작 완료")

    yield

    # 종료 시 실행
    logger.info("🔄 [LIFESPAN] 애플리케이션 종료 중...")


# 먼저 lifespan 함수가 제대로 정의되었는지 확인
logger.info(f"Lifespan function defined: {lifespan}")


# FastAPI 앱 인스턴스 생성 (Swagger UI 문서에 표시될 메타데이터 포함)
app = FastAPI(
    title="TUNING API",
    description="조직 내부 사용자 간의 자연스럽고 부담 없는 소통을 돕는 소셜 매칭 서비스 API",
    version="1.0.0"
    #lifespan=lifespan,
)
register_exception_handlers(app)  # 반드시 포함

# 라우터 등록 - API를 기능별로 모듈화
app.include_router(HealthRouter().router)
app.include_router(UserRouter().router)
app.include_router(UserRouter().router_v1)
app.include_router(UserRouter().router_v2)
app.include_router(UserRouter().router_v3)
app.include_router(TuningRouter().router_v1)
app.include_router(TuningRouter().router_v3)
app.include_router(PerformanceRouter().router)


# 루트 경로 핸들러 - 개발 환경에서는 API 문서(Swagger)로 리다이렉트, 프로덕션에서는 접근 제한
@app.get("/")
async def root():
    env = os.getenv("ENVIRONMENT")
    if env == "dev":
        return RedirectResponse(url="/docs")
    else:
        return HTTPException(status_code=404)
