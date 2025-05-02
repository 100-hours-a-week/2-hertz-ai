# utils/error_handler.py
import traceback
from typing import Dict, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from utils.logger import logger


def register_exception_handlers(app: FastAPI) -> None:
    """
    FastAPI 애플리케이션에 전역 예외 핸들러 등록

    Args:
        app: FastAPI 애플리케이션 인스턴스
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """HTTP 예외 핸들러"""
        # HTTP 예외에 detail이 Dict 형태로 들어있으면 그대로 사용
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)

        # 기본 HTTP 예외 처리
        error_code = _status_to_error_code(exc.status_code)
        logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code, content={"code": error_code, "data": None}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """요청 검증 예외 핸들러"""
        error_details = []
        for error in exc.errors():
            loc = " -> ".join(str(item) for item in error["loc"])
            error_details.append(f"{loc}: {error['msg']}")

        error_message = ", ".join(error_details)
        logger.error(f"Validation Error: {error_message}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": "BAD_REQUEST_VALIDATION_ERROR",
                "data": {"errors": error_details},
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """모든 예외에 대한 기본 핸들러"""
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc()

        # 로그에 예외 스택 트레이스 기록
        logger.error(
            f"Unhandled {error_type}: {error_message}\n"
            f"Path: {request.url.path}\n"
            f"Traceback: {error_traceback}"
        )

        # 프로덕션 환경에서는 상세 오류를 노출하지 않음
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"code": "INTERNAL_SERVER_ERROR", "data": None},
        )


def _status_to_error_code(status_code: int) -> str:
    """
    HTTP 상태 코드를 에러 코드 문자열로 변환

    Args:
        status_code: HTTP 상태 코드

    Returns:
        에러 코드 문자열
    """
    if status_code == 400:
        return "BAD_REQUEST"
    elif status_code == 401:
        return "UNAUTHORIZED"
    elif status_code == 403:
        return "FORBIDDEN"
    elif status_code == 404:
        return "NOT_FOUND"
    elif status_code == 409:
        return "CONFLICT"
    elif status_code == 422:
        return "UNPROCESSABLE_ENTITY"
    elif status_code == 429:
        return "TOO_MANY_REQUESTS"
    elif status_code >= 500:
        return "INTERNAL_SERVER_ERROR"
    else:
        return f"ERROR_{status_code}"


def format_error_response(
    error_code: str, message: Optional[str] = None, data: Optional[Dict] = None
) -> Dict[str, Union[str, Dict, None]]:
    """
    표준화된 에러 응답 생성

    Args:
        error_code: 에러 코드
        message: 에러 메시지 (선택)
        data: 추가 데이터 (선택)

    Returns:
        표준화된 에러 응답 딕셔너리
    """
    response = {"code": error_code, "data": data if data is not None else None}

    if message:
        if response["data"] is None:
            response["data"] = {}
        response["data"]["message"] = message

    return response
