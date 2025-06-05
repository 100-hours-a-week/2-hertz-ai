import logging
import os

import chromadb

chroma_client = None


def is_client_alive(client):
    try:
        client.list_collections()  # 헬스체크
        return True
    except Exception as e:
        logging.warning(f"[Chroma] 클라이언트 응답 없음: {e}")
        print(f"ChromaDB 클라이언트 헬스체크 실패: {e}")
        return False


def get_chroma_client():
    global chroma_client

    if chroma_client is not None and is_client_alive(chroma_client):
        return chroma_client
    else:
        print("ChromaDB 클라이언트 재연결 시도")
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
            logging.error("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")
            raise RuntimeError("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")

        return chroma_client
    except Exception as e:
        logging.exception(f"[Chroma] 클라이언트 초기화 실패: {e}")
        chroma_client = None
        return None
