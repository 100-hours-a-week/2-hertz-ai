import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from utils.logger import logger

MODEL_NAME = "jhgan/ko-sbert-sts"
MODEL_DIR_NAME = MODEL_NAME.replace("/", "-")

# 루트 디렉토리 기준으로 고정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# KCELECTRA_BASE_HOME 환경변수 있으면 사용, 없으면 model-cache 기본 경로
MODEL_CACHE = os.environ.get(
    "SENTENCE_TRANSFORMERS_HOME", os.path.join(BASE_DIR, "model-cache")
)

# 모델 최종 경로
MODEL_PATH = Path(MODEL_CACHE) / MODEL_DIR_NAME

os.makedirs(MODEL_CACHE, exist_ok=True)


def _download_model():

    # 모델 디렉토리에 모델 가중치 파일 등이 이미 존재하는지 체크
    if os.path.isdir(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
        logger.info(
            f"[INFO] '{MODEL_NAME}' 모델이 이미 존재합니다: {MODEL_PATH} (다운로드 건너뜀)"
        )
    else:
        try:
            logger.info(
                f"[INFO] '{MODEL_NAME}' 모델을 다운로드합니다 (경로: {MODEL_PATH})..."
            )
            model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
            model.save(str(MODEL_PATH))

            logger.info(f"[INFO] 모델 다운로드 및 저장 완료: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"[ERROR] 모델 다운로드 또는 저장 중 오류 발생: {e}")
            raise


if __name__ == "__main__":
    _download_model()
