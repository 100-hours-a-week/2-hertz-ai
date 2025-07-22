# 에러 처리 유틸리티
# utils/error_handler.py
import traceback
from typing import Dict, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException as FastAPIHTTPException
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

        if isinstance(exc.detail, list):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": _status_to_error_code(exc.status_code),
                        "message": "요청 필드 오류",
                        "details": [
                            {
                                "field": ".".join(str(x) for x in e.get("loc", [])),
                                "reason": e.get("msg", ""),
                                "type": e.get("type", ""),
                            }
                            for e in exc.detail
                        ],
                    }
                },
            )

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

        logger.error(f"Validation Error: {exc.errors()}")

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "CENSOR_BAD_REQUEST_VALIDATION_ERROR",
                    "message": "필수 필드 누락 또는 형식 오류",
                    "details": [
                        {
                            "field": ".".join(str(x) for x in err["loc"]),
                            "reason": err["msg"],
                            "type": err["type"],
                        }
                        for err in exc.errors()
                    ],
                }
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
            content={"code": "CENSORED_INTERNAL_SERVER_ERROR", "data": None},
        )

    # 추가: FastAPI의 예외도 별도로 등록
    @app.exception_handler(FastAPIHTTPException)
    async def fastapi_http_exception_handler(
        request: Request, exc: FastAPIHTTPException
    ) -> JSONResponse:
        # 기존 Starlette 예외 핸들러와 동일하게 처리
        return await http_exception_handler(request, exc)


def _status_to_error_code(status_code: int) -> str:
    """
    HTTP 상태 코드를 에러 코드 문자열로 변환

    Args:
        status_code: HTTP 상태 코드

    Returns:
        에러 코드 문자열
    """
    if status_code == 400:
        return "CENSORED_BAD_REQUEST"
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
        return "CENSORED_INTERNAL_SERVER_ERROR"
    else:
        return f"ERROR_{status_code}"


def format_error_response(
    error_code: str,
    message: Optional[str] = None,
    data: Optional[Dict] = None,
) -> Dict[str, Union[str, Dict, None]]:
    """
    표준화된 에러 응답 딕셔너리 생성
    """
    response = {"code": error_code, "data": data if data is not None else None}

    if message:
        if response["data"] is None:
            response["data"] = {}
        response["data"]["message"] = message

    return response  # ❗ 딕셔너리만 반환
