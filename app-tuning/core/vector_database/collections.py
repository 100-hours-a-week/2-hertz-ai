import logging

from .client import get_chroma_client

_user_collection = None
_similarity_collection = None

USER_COLLECTION_NAME = "user_profiles"
SIMILARITY_COLLECTION_NAME = "user_similarities"


def _is_alive(collection) -> bool:
    try:
        collection.count()  # 헬스 체크
        return True
    except Exception as e:
        logging.debug(f"[Chroma] 컬렉션 응답 없음: {e}")
        return False


_collection_cache = {}


def _get_or_create_collection(cache_key, collection_name):
    collection = _collection_cache.get(cache_key)

    if collection:
        if _is_alive(collection):
            return collection
        else:
            logging.warning(
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


def get_similarity_collection():
    return _get_or_create_collection("similarity", SIMILARITY_COLLECTION_NAME)


#  ChromaDB 데이터베이스 컬렉션 삭제 후, 재생성(테스트서버 초기화용)
def reset_collections():
    """
    로컬 및 서버모드 모두에서 컬렉션을 재생성 및 전역 변수 초기화(내부 테스트 환경용)
    (주의: 모든 데이터가 삭제됨)
    """
    try:
        client = get_chroma_client()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        # 삭제
        client.delete_collection(USER_COLLECTION_NAME)
        client.delete_collection(SIMILARITY_COLLECTION_NAME)

        # 전역 캐시 초기화
        global _user_collection, _similarity_collection
        _user_collection = client.get_or_create_collection(USER_COLLECTION_NAME)
        _similarity_collection = client.get_or_create_collection(
            SIMILARITY_COLLECTION_NAME
        )

    except Exception as e:
        raise RuntimeError(f"ChromaDB 컬렉션 초기화 실패: {e}")
