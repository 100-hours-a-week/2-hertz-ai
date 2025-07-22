"""
매칭 추천 기능을 담당하는 컨트롤러
사용자 간 유사도 기반 매칭을 처리하고 결과를 반환
"""

from fastapi import HTTPException
from schemas.tuning_schema import TuningMatchingList, TuningResponse
from services.tuning_service import get_matching_users, get_matching_users_by_category
from utils.logger import logging

logger = logging.getLogger(__name__)


async def get_tuning_matches(user_id: int) -> TuningResponse:
    """
    사용자 ID를 기반으로 매칭 추천을 제공하는 컨트롤러 함수

    Args:
        userId: 매칭을 요청한 사용자의 ID

    Returns:
        Dictionary containing the response code and matching user IDs list

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    user_id = str(user_id)
    try:
        result = await get_matching_users(user_id)

        if not result:
            return {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in tuning controller: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TuningResponse(
                code="TUNING_INTERNAL_SERVER_ERROR", data=None
            ).model_dump(),
        )

    return TuningResponse(
        code="TUNING_SUCCESS", data=TuningMatchingList(userIdList=result)
    )


async def get_tuning_matches_by_category(user_id: int, category: str) -> TuningResponse:
    """
    사용자 ID를 기반으로 매칭 추천을 제공하는 컨트롤러 함수

    Args:
        userId(int): 추천을 요청한 사용자의 ID
        category(Optional[str]): 매칭 카테고리 ("friend", "couple")

    Returns:
        Dictionary containing the response code and matching user IDs list

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    user_id = str(user_id)
    try:
        result = await get_matching_users_by_category(user_id, category)

        if not result:
            return TuningResponse(code="TUNING_SUCCESS_BUT_NO_MATCH", data=None)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=TuningResponse(
                code="TUNING_INTERNAL_SERVER_ERROR", data=None
            ).model_dump(),
        )

    return TuningResponse(
        code="TUNING_SUCCESS", data=TuningMatchingList(userIdList=result)
    )
