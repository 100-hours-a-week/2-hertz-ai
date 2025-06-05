import json

import numpy as np

# from app.core.embedding import convert_user_to_text, embed_fields
from core.embedding import convert_user_to_text, embed_fields_optimized
from core.enum_process import convert_to_korean

# from app.core.matching_score import compute_matching_score
from core.matching_score_optimized import compute_matching_score_optimized
from core.vector_database import (
    clean_up_similarity,
    delete_user,
    get_similarity_collection,
    get_user_collection,
)
from fastapi import HTTPException

# from app.models.sbert_loader import model
from models.sbert_loader import get_model
from schemas.user_schema import EmbeddingRegister
from utils import logger


# 메타데이터 저장 시 문자열로 반환하기 위함
def safe_join(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)


@logger.log_performance(operation_name="prepare_embedding_data", include_memory=True)
def prepare_embedding_data(
    user_dict: dict, target_fields: list[str]
) -> tuple[list[float], dict]:
    """
    사용자 딕셔너리에서 임베딩 및 메타데이터를 생성

    Args:
        user_dict: 사용자 정보 딕셔너리
        target_fields: 임베딩에 사용할 필드 목록

    Returns:
        Tuple of (embedding vector, metadata dict)
    """
    try:
        user_dict = convert_to_korean(user_dict)  # 한글화 처리

        user_text = convert_user_to_text(user_dict, target_fields)

        model = get_model()
        embedding = model.encode(user_text).tolist()  # 통합 텍스트 임베딩 생성

        # field_embeddings = embed_fields(user_dict, target_fields, model=model)
        field_embeddings = embed_fields_optimized(user_dict, target_fields)

        metadata = {k: safe_join(v) for k, v in user_dict.items()}
        metadata["field_embeddings"] = json.dumps(field_embeddings)

        return embedding, metadata

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "EMBEDDING_PREPARATION_FAILED", "message": str(e)},
        )


# 매칭 스코어 정보 DB 저장
def upsert_similarity(user_id: str, embedding: list, similarities: dict):
    get_similarity_collection().upsert(
        ids=[user_id],
        embeddings=[embedding],
        metadatas=[{"userId": user_id, "similarities": json.dumps(similarities)}],
    )


# 매칭 스코어 정보 역방향 DB 저장
@logger.log_performance(
    operation_name="update_reverse_similarities", include_memory=True
)
def update_reverse_similarities(user_id: str, similarities: dict):
    for other_id, score in similarities.items():
        try:
            other_id = str(other_id)
            # 1. similarity_collection에서 상대방 데이터 조회
            other_sim = get_similarity_collection().get(
                ids=[other_id], include=["metadatas", "embeddings"]
            )

            if not other_sim or not other_sim.get("metadatas"):
                # 2. 없으면 user_profiles에서 embedding만 가져옴
                other_user = get_user_collection().get(
                    ids=[other_id], include=["embeddings"]
                )
                other_embedding = other_user["embeddings"][0]
                reverse_map = {user_id: score}
            else:
                other_meta = other_sim["metadatas"][0]
                other_embedding = other_sim["embeddings"][0]
                try:
                    reverse_map = json.loads(other_meta.get("similarities", "{}"))
                except json.JSONDecodeError:
                    reverse_map = {}

                # 3. 값이 바뀐 경우에만 업데이트
                if reverse_map.get(user_id) == score:
                    continue

                reverse_map[user_id] = score

            # 4. 실제 벡터 등록/업데이트
            upsert_similarity(other_id, other_embedding, reverse_map)

        except Exception as e:
            print(f"[REVERSE_SIMILARITY_UPDATE_ERROR]: {other_id} / {e}")
            raise RuntimeError(f"역방향 유사도 업데이트 실패: {e}")


# 현재 유저가 저장하지 않은 상대방의 기존 유사도를 병합
@logger.log_performance(
    operation_name="enrich_with_reverse_similarities", include_memory=True
)
def enrich_with_reverse_similarities(
    user_id: str, similarities: dict, all_users: dict
) -> dict:
    updated_map = dict(similarities)

    for other_id in all_users["ids"]:
        if str(other_id) == user_id:
            continue

        other_sim = get_similarity_collection().get(ids=[other_id])
        if not other_sim or not other_sim.get("metadatas"):
            continue

        other_meta = other_sim["metadatas"][0]
        other_map = json.loads(other_meta.get("similarities", "{}"))

        if user_id in other_map and other_id not in updated_map:
            updated_map[other_id] = other_map[user_id]

    return updated_map


# 전체 유저와의 매칭 스코어 계산 및 저장
@logger.log_performance(
    operation_name="update_similarity_for_users", include_memory=True
)
def update_similarity_for_users(user_id: str) -> dict:
    try:
        all_users = get_user_collection().get(include=["embeddings", "metadatas"])
        ids, embeddings, metadatas = (
            all_users["ids"],
            all_users["embeddings"],
            all_users["metadatas"],
        )

        if user_id not in ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SIMILARITY_USER_NOT_FOUND",
                    "message": f"User ID {user_id} not found",
                },
            )
        idx = ids.index(user_id)
        user_embedding, user_meta = embeddings[idx], metadatas[idx]

        # similarities = compute_matching_score(
        #     user_id=user_id,
        #     user_embedding=user_embedding,
        #     user_meta=user_meta,
        #     all_users=all_users,
        # )

        similarities = compute_matching_score_optimized(
            user_id=user_id,
            user_embedding=user_embedding,
            user_meta=user_meta,
            all_users=all_users,
        )

        # 현재 유저 유사도 저장
        upsert_similarity(user_id, user_embedding, similarities)

        # 역방향 저장
        update_reverse_similarities(user_id, similarities)

        # 반대방향에도 user_id가 존재하는 경우 통합
        updated_map = enrich_with_reverse_similarities(user_id, similarities, all_users)

        # 최종 반영
        upsert_similarity(user_id, user_embedding, updated_map)

        return {"userId": user_id, "updated_similarities": len(updated_map)}

    except HTTPException as http_ex:
        raise http_ex

    except Exception as e:
        print(f"[SIMILARITY_UPDATE_ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SIMILARITY_UPDATE_FAILED",
                "message": str(e),
            },
        )


# 필드 검증 로직 함수
def validate_user_fields(user: EmbeddingRegister) -> None:
    required_fields = ["MBTI", "religion", "smoking", "drinking"]
    missing_required = [
        field for field in required_fields if not getattr(user, field, None)
    ]
    if missing_required:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", field],
                    "msg": "At least one item is required",
                    "type": "value_error.min_items",
                }
                for field in missing_required
            ],
        )

    min_required_fields = [
        "personality",
        "preferredPeople",
        "currentInterests",
        "favoriteFoods",
        "likedSports",
        "pets",
        "selfDevelopment",
    ]
    empty_fields = [
        field
        for field in min_required_fields
        if not getattr(user, field, None) or len(getattr(user, field)) < 1
    ]
    if empty_fields:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", field],
                    "msg": "At least one item is required",
                    "type": "value_error.min_items",
                }
                for field in empty_fields
            ],
        )


# 아이디  중복 검사
def check_duplicate_user(user_id: str) -> None:
    existing = get_user_collection().get(ids=[user_id])
    if existing and user_id in existing.get("ids", []):
        raise HTTPException(
            status_code=409,
            detail={"code": "EMBEDDING_CONFLICT_DUPLICATE_ID", "data": None},
        )


# 신규 유저 등록과 매칭 스코어 계산 처리 통합 로직
@logger.log_performance(operation_name="register_user", include_memory=True)
async def register_user(user: EmbeddingRegister) -> None:
    # start_time = time.time()

    try:
        user_id = str(user.userId)
        validate_user_fields(user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_REGISTER_VALIDATION_FAILED",
                "message": str(e),
            },
        )

    check_duplicate_user(user_id)

    try:
        user_dict = user.model_dump()

        target_fields = [
            "emailDomain",
            "gender",
            "religion",
            "smoking",
            "drinking",
            "currentInterests",
            "favoriteFoods",
            "likedSports",
            "pets",
            "selfDevelopment",
            "hobbies",
        ]

        embedding, metadata = prepare_embedding_data(user_dict, target_fields)

        get_user_collection().add(
            ids=[user_id], embeddings=[embedding], metadatas=[metadata]
        )

    except Exception as e:
        print(f"[ REGISTER ERROR] 사용자 등록 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "EMBEDDING_REGISTER_SERVER_ERROR", "message": str(e)},
        )

    try:
        update_similarity_for_users(user_id)
    except Exception as e:
        print(f"[ SIMILARITY ERROR] 유사도 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_REGISTER_SIMILARITY_UPDATE_FAILED",
                "message": str(e),
            },
        )

    # elapsed = round(time.time() - start_time, 3)

    # return {
    # "status": "registered",
    # "userId": user_id,
    # "matchedUserCount": similarity_result.get("updated_similarities", 0),
    # "time_taken_seconds": elapsed}


# 전체 유저와의 매칭 스코어 계산 및 저장
@logger.log_performance(operation_name="delete_user", include_memory=True)
def delete_user_metatdata(user_id: int):
    try:
        clean_up_similarity(user_id)
        delete_user(user_id)
        return {"code": "EMBEDDING_DELETE_SUCCESS", "data": None}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_DELETE_SERVER_ERROR",
                "message": str(e),
            },
        )
