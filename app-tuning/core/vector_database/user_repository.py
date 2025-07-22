from fastapi import HTTPException
from utils.logger import logger

from .collections import get_similarity_collection, get_user_collection


def get_user_data(user_id: str):
    """
    특정 사용자 ID에 해당하는 메타데이터 조회
    """
    try:
        collection = get_user_collection()
        result = collection.get(ids=[user_id], include=["metadatas"])
        if not result["metadatas"] or result["metadatas"][0] is None:
            raise HTTPException(
                status_code=404, detail="사용자 정보를 찾을 수 없습니다."
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_users_data(user_ids: list[str]):
    """
    여러 사용자 ID에 대한 메타데이터 조회
    """
    collection = get_user_collection()
    return collection.get(ids=user_ids, include=["metadatas"])


async def list_users():
    """
    전체 사용자 목록 조회
    """
    collection = get_user_collection()
    return collection.get()


def delete_user(user_id: int):

    user_id = str(user_id)
    user_collection = get_user_collection()
    similarity_collection = get_similarity_collection()

    existing = user_collection.get(ids=[user_id])
    if not existing or user_id not in existing.get("ids", []):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EMBEDDING_DELETE_NOT_FOUND_USER",
                "data": f"User ID '{user_id}' not found in user_profiles",
            },
        )

    try:
        user_collection.delete(ids=[user_id])
        similarity_collection.delete(ids=[user_id])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_DELETE_SERVER_ERROR",
                "message": f"Failed to delete user '{user_id}': {str(e)}",
            },
        )


# ----------v3-------------


def delete_user_v3(user_id: int):

    user_id = str(user_id)
    user_collection = get_user_collection()

    existing = user_collection.get(ids=[user_id])
    if not existing or user_id not in existing.get("ids", []):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EMBEDDING_DELETE_NOT_FOUND_USER",
                "data": f"User ID '{user_id}' not found in user_profiles",
            },
        )

    try:
        user_collection.delete(ids=[user_id])

        # similarity 관련 모든 컬렉션에서 삭제
        for category in ["friend", "couple"]:
            similarity_collection = get_similarity_collection(category)
            similarity_collection.delete(ids=[user_id])

        logger.info(
            f"user_id '{user_id}' 삭제 완료 (user_profiles 및 모든 similarity 컬렉션)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_DELETE_SERVER_ERROR",
                "message": f"Failed to delete user '{user_id}': {str(e)}",
            },
        )
