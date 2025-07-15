import asyncio
import os

import requests
from utils.logger import logger

import chromadb

chroma_client = None


def is_client_alive(client):
    try:
        client.list_collections()  # 헬스체크
        return True
    except Exception as e:
        logger.warning(f"[Chroma] 클라이언트 응답 없음: {e}")
        logger.error(f"ChromaDB 클라이언트 헬스체크 실패: {e}")
        return False


async def wait_for_chromadb():
    """ChromaDB Health Check API를 통해 연결 상태 확인"""
    chromadb_host = os.getenv("CHROMA_HOST", "localhost")
    chromadb_port = os.getenv("CHROMA_PORT", "8001")
    health_url = f"http://{chromadb_host}:{chromadb_port}/api/v1/heartbeat"

    max_retries = 5
    retry_delay = 2

    logger.info(f"ChromaDB Health Check 시작: {health_url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "UP":
                    logger.info(
                        f"ChromaDB 연결 성공! Collections: {data.get('collections_count', 'N/A')}"
                    )
                    return True
            logger.info(f"ChromaDB 상태 확인 중... 시도 {attempt + 1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            logger.info(f"ChromaDB 연결 시도 {attempt + 1}/{max_retries} 실패: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)

    logger.error("ChromaDB 연결 실패 - 최대 재시도 횟수 초과")
    return False


def get_chroma_client():
    global chroma_client

    if chroma_client is not None and is_client_alive(chroma_client):
        return chroma_client
    else:
        logger.info("ChromaDB 클라이언트 연결 시도")
        chroma_client = None  # 죽은 연결 무효화

    try:
        mode = os.getenv("CHROMA_MODE", "server")

        if mode == "local":
            # 로컬 PersistentClient 사용
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            chroma_path = os.path.join(base_dir, "chroma_db")
            chroma_client = chromadb.PersistentClient(path=chroma_path)
        else:
            # 서버 모드 (기본)
            host = os.getenv("CHROMA_HOST", "localhost")
            port = int(os.getenv("CHROMA_PORT", "8001"))

            host = host.replace("http://", "").replace("https://", "")

            chroma_client = chromadb.HttpClient(host=host, port=port)

        if not is_client_alive(chroma_client):
            logger.error("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")
            raise RuntimeError("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")

        return chroma_client
    except Exception as e:
        logger.exception(f"[Chroma] 클라이언트 초기화 실패: {e}")
        chroma_client = None
        return None
