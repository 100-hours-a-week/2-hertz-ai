import json
from typing import Optional

from core.vector_database import (  # user_couple_similarities,; get_user_friend_similarities,
    get_similarities,
    get_users_data,
)
from fastapi import HTTPException
from schemas.tuning_schema import TuningResponse
from utils import logger

SIM_COLLECTIONS = {
    None: "user_similarities",
    "friend": "friend_similarities",
    "couple": "couple_similarities",
}


# 유사도 데이터를 가져오고 파싱하는 함수
async def fetch_user_similarities(
    user_id: str, category: Optional[str] = None
) -> dict[str, float]:
    similarities_data = None
    similarities_data = await get_similarities(user_id, category)
    # 유사도 데이터가 없으면 404 에러 반환
    if not similarities_data or not similarities_data.get("metadatas"):
        raise HTTPException(
            status_code=404, detail={"code": "TUNING_NOT_FOUND_USER", "data": None}
        )

    try:
        # similarities 필드에서 JSON 문자열을 딕셔너리로 파싱
        similarity_map = json.loads(
            similarities_data["metadatas"][0].get("similarities", "{}")
        )

        # 문자열 userId → float 유사도 점수 형태로 변환
        return {str(k): float(v) for k, v in similarity_map.items()}
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "INVALID_SIMILARITY_DATA", "message": str(e)},
        )


# userId 리스트를 받아 해당 유저들의 메타데이터를 가져오는 함수
async def fetch_users_metadata(user_ids: list[str]) -> dict[str, dict]:

    if not user_ids:
        return {}

    # user_ids가 리스트가 아닐 경우 방어
    if not isinstance(user_ids, list):
        raise ValueError(f"Expected user_ids to be a list, got {type(user_ids)}")
    try:
        user_data = await get_users_data(user_ids)

        # userId를 키로 하는 metadata 딕셔너리로 변환
        return {
            meta["userId"]: meta
            for meta in user_data.get("metadatas", [])
            if isinstance(meta, dict) and "userId" in meta
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "USER_METADATA_FETCH_FAILED", "message": str(e)},
        )


# 유사도 정보와 메타데이터를 기반으로 추천 ID만 추출하는 함수
def format_recommendations(
    similarities: dict[str, float],
    metadata: dict[str, dict],
    top_k: int = 100,
    category: Optional[str] = None,
) -> list[int]:

    # 유사도 점수를 기준으로 내림차순 정렬 후 상위 N개만 추출
    sorted_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    # metadata가 존재하는 유저만 ID로 반환
    return [int(uid) for uid, _ in sorted_users if uid in metadata]


# 카테고리별(couple, friend) 추천 결과를 반환하는 메인 함수 (category 값 로깅되는지 확인하기 )
@logger.log_performance(operation_name="get_matching_users", include_memory=True)
async def get_matching_users(
    user_id: str, category: Optional[str] = None
) -> TuningResponse:
    """
    친구/커플 추천 결과를 반환하는 통합 함수
    """
    category = category or "friend"  # 기본값 설정
    logger.info(f"[CATEGORY] 추천 요청 category: {category}")
    try:
        # 카테고리에 따라 컬렉션 선택
        similarities = await fetch_user_similarities(user_id, category)

        user_ids = list(similarities.keys())
        metadata = await fetch_users_metadata(user_ids)

        recommended_ids = format_recommendations(
            similarities, metadata, category=category
        )

        return TuningResponse(
            code="TUNING_SUCCESS" if recommended_ids else "TUNING_SUCCESS_BUT_NO_MATCH",
            data=None,
        )

    except Exception:
        raise HTTPException(status_code=500, detail="TUNING_INTERNAL_SERVER_ERROR")
