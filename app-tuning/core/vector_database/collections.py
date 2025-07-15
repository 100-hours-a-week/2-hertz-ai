from typing import Optional

from utils.logger import logger

from .client import get_chroma_client

USER_COLLECTION_NAME = "user_profiles"
SIMILARITY_COLLECTION_NAME = "user_similarities"
FRIEND_SIMILARITY_COLLECTION_NAME = "friend_similarities"
COUPLE_SIMILARITY_COLLECTION_NAME = "couple_similarities"
COLLECTION_MAP = {
    None: ("user_similarity", SIMILARITY_COLLECTION_NAME),
    "friend": ("friend_similarity", FRIEND_SIMILARITY_COLLECTION_NAME),
    "couple": ("couple_similarity", COUPLE_SIMILARITY_COLLECTION_NAME),
}


def _is_alive(collection) -> bool:
    try:
        collection.count()  # 헬스 체크
        return True
    except Exception as e:
        logger.debug(f"[Chroma] 컬렉션 응답 없음: {e}")
        return False


_collection_cache = {}


def _get_or_create_collection(cache_key, collection_name):
    collection = _collection_cache.get(cache_key)

    if collection:
        if _is_alive(collection):
            return collection
        else:
            logger.warning(
                f"[Chroma] 컬렉션 '{collection_name}' 연결이 끊어졌습니다. 재연결 시도 중..."
            )
            _collection_cache[cache_key] = None  # 무효화

    client = get_chroma_client()
    if client is None:
        raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

    try:
        collection = client.get_or_create_collection(collection_name)
        _collection_cache[cache_key] = collection
        return collection
    except Exception as e:
        raise RuntimeError(f"{collection_name} 컬렉션 초기화 실패: {e}") from e


def get_user_collection():
    return _get_or_create_collection("user", USER_COLLECTION_NAME)


def get_similarity_collection(category: Optional[str] = None):
    """
    카테고리에 따라 적절한 similarity 컬렉션을 반환합니다.

    Args:
        category (str): 'friend', 'couple' 또는 None (기본)

    Returns:
        Collection 객체
    """
    # 'category'가 COLLECTION_MAP의 유효한 키인지 확인
    if category not in COLLECTION_MAP:
        # 유효하지 않은 경우
        raise ValueError(
            f"Invalid category: '{category}'. Allowed values are 'friend', 'couple', or None."
        )

    key, name = COLLECTION_MAP[category]
    return _get_or_create_collection(key, name)


#  ChromaDB 데이터베이스 컬렉션 삭제 후, 재생성(테스트서버 초기화용)
def reset_collections():
    """
    ChromaDB 컬렉션 초기화 (테스트 서버 전용)
    모든 컬렉션을 삭제하고 재생성하며 전역 변수 초기화까지 수행합니다.
    ⚠️ 모든 벡터/메타데이터가 삭제되므로 주의해서 사용하세요.
    """
    try:
        client = get_chroma_client()
        _collection_cache.clear()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        # 컬렉션 이름 → 전역 변수 매핑
        collection_map = {
            USER_COLLECTION_NAME: "_user_collection",
            FRIEND_SIMILARITY_COLLECTION_NAME: "_friend_similarity_collection",
            COUPLE_SIMILARITY_COLLECTION_NAME: "_couple_similarity_collection",
            SIMILARITY_COLLECTION_NAME: "_similarity_collection",
        }

        # 삭제 + 재생성 + 전역 초기화
        for collection_name, global_var in collection_map.items():
            try:
                client.delete_collection(collection_name)
            except Exception as e:
                logger.error(f"⚠️ 컬렉션 삭제 실패 [{collection_name}]: {e}")

            collection = client.get_or_create_collection(collection_name)
            globals()[global_var] = collection  # 전역 변수 초기화 (제거 여부 확인 필요)

        logger.info("✅ 모든 컬렉션 초기화 및 전역 변수 설정 완료")

    except Exception as e:
        raise RuntimeError(f"❌ ChromaDB 초기화 실패: {e}")
