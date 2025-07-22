"""
사용자 등록 및 임베딩 벡터 생성을 담당하는 컨트롤러
라우터에서 받은 요청을 처리하고 서비스 레이어와 연결
"""

import logging

from core.vector_database import (
    get_similarities,
    list_similarities,
    list_users,
    reset_collections,
)
from fastapi import HTTPException
from schemas.user_schema import BaseResponse, EmbeddingRegister
from services.user_service import (
    delete_user_metatdata,
    delete_user_metatdata_v3,
    register_user,
    register_user_v3,
)

logger = logging.getLogger(__name__)


async def db_user_list():
    result = await list_users()
    return {
        "code": "REGISTERD_ID_CHECKED",
        "data": {
            "ids": result.get("ids", []),
            "count": len(result.get("ids", [])),
            # "metadatas": result.get("metadatas", []),
        },
    }


async def db_similarity_list():
    result = await list_similarities()
    return {
        "code": "REGISTERD_SIMILARITY_CHECKED",
        "data": {
            "ids": result.get("ids", []),
            "count": len(result.get("ids", [])),
            "metadatas": (result.get("metadatas", [])),
        },
    }


async def db_similarity_list_v3(category: str):
    result = await get_similarities(category)
    return {
        "code": "REGISTERD_SIMILARITY_CHECKED",
        "data": {
            "collection_name": category,
            "ids": result.get("ids", []),
            "count": len(result.get("ids", [])),
            "metadatas": (result.get("metadatas", [])),
        },
    }


async def db_reset_data():
    reset_collections()
    return BaseResponse(status="success", code="CHROMADB_RESET_SUCCESS")


async def create_user_v3(user_data: EmbeddingRegister) -> BaseResponse:
    try:
        await register_user_v3(user_data)
        return BaseResponse(code="EMBEDDING_REGISTER_SUCCESS", data=None)
    except HTTPException as http_ex:
        logger.warning(f"[REGISTER_USER_HTTP_ERROR] {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(
            f"[REGISTER_USER_FATAL_ERROR]: {str(e)}"
        )  # 자동 traceback 포함
        raise HTTPException(
            status_code=500,
            detail=BaseResponse(
                code="EMBEDDING_REGISTER_SERVER_ERROR", data=None
            ).model_dump(),
        )


async def delete_user_data_v3(user_id: int) -> BaseResponse:
    try:
        delete_user_metatdata_v3(user_id)
        return BaseResponse(code="EMBEDDING_DELETE_SUCCESS", data=None)
    except HTTPException as http_ex:
        logger.warning(f"[EMBEDDING_DELETE_HTTP_ERROR] {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(f"[EMBEDDING_DELETE_FATAL_ERROR]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=BaseResponse(
                code="EMBEDDING_DELETE_SERVER_ERROR", data=None
            ).model_dump(),
        )


# -------------------- 아래는 기존 버전----------------------


async def create_user(user_data: EmbeddingRegister) -> BaseResponse:
    """
    새 사용자를 등록하고 임베딩 벡터를 생성하는 컨트롤러 함수

    Args:
        user_data: 사용자 등록 데이터 (Pydantic 모델)

    Returns:
        Dictionary containing the response code and result

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    try:
        await register_user(user_data)
        return BaseResponse(code="EMBEDDING_REGISTER_SUCCESS", data=None)
    except HTTPException as http_ex:
        logger.warning(f"[REGISTER_USER_HTTP_ERROR] {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(
            f"[REGISTER_USER_FATAL_ERROR]: {str(e)}"
        )  # 자동 traceback 포함
        raise HTTPException(
            status_code=500,
            detail=BaseResponse(
                code="EMBEDDING_REGISTER_SERVER_ERROR", data=None
            ).model_dump(),
        )


async def delete_user_data(user_id: int) -> BaseResponse:
    """
    사용자 데이터를 삭제하는 컨트롤러 함수

    Args:
        user_id (int): 삭제할 사용자의 ID

    Returns:
        dict: 응답 코드와 결과 데이터를 포함한 딕셔너리

    Raises:
        HTTPException: 사용자 데이터가 없거나 서버 오류가 발생한 경우
    """
    try:
        delete_user_metatdata(user_id)
        return BaseResponse(code="EMBEDDING_DELETE_SUCCESS", data=None)
    except HTTPException as http_ex:
        logger.warning(f"[EMBEDDING_DELETE_HTTP_ERROR] {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(f"[EMBEDDING_DELETE_FATAL_ERROR]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=BaseResponse(
                code="EMBEDDING_DELETE_SERVER_ERROR", data=None
            ).model_dump(),
        )
