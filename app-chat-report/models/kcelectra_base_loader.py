"""
SBERT 모델 로더 모듈
한국어 문장 임베딩을 위한 SBERT 모델을 효율적으로 로드하고 관리
싱글톤 패턴을 적용하여 메모리 사용량 최적화 및 일관된 추론 환경 제공
"""

import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from utils.logger import logger


# 모듈 임포트 시점에 바로 모델 초기화
def _load_model():
    # CPU 스레드 수 최적화 - 시스템의 모든 코어 활용
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # 최소 1개는 사용하도록 보장

    # 모델 로드

    # 기존 코드
    # loaded_model = SentenceTransformer("jhgan/ko-sbert-nli")

    # 환경변수에서 모델 경로 가져오기

    MODEL_NAME = "jinkyeongk/kcELECTRA-toxic-detector"
    MODEL_DIR_NAME = MODEL_NAME.replace("/", "-")

    # app-tuning 디렉토리 기준으로 고정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # KCELECTRA_BASE_HOME 환경변수 있으면 사용, 없으면 model-cache 기본 경로
    MODEL_CACHE = os.environ.get(
        "KCELECTRA_BASE_HOME", os.path.join(BASE_DIR, "model-cache")
    )

    # 모델 최종 경로
    MODEL_PATH = Path(MODEL_CACHE) / MODEL_DIR_NAME

    # 모델 경로가 존재하지 않으면 다운로드 시도
    if not MODEL_PATH.exists():
        print(f"모델이 존재하지 않습니다: {MODEL_PATH}")
        print("모델 다운로드를 시도합니다...")

        try:
            subprocess.run(["python", "scripts/download_model.py"], check=True)
            logger.info("모델 다운로드가 완료되었습니다.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "모델 다운로드에 실패했습니다. 스크립트를 수동으로 실행해주세요: "
                "'python scripts/download_model.py'"
            ) from e

        # 다운로드 후 다시 존재하는지 확인
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"모델 다운로드 후에도 경로에 모델이 존재하지 않습니다: {MODEL_PATH}"
            )

    try:
        # 모델 및 토크나이저 로드
        logger.info(f"모델 로딩 중: {MODEL_PATH}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        logger.info("모델 로딩 완료")

        # pipeline 구성
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # 예열 (간단한 추론 한번 수행)
        logger.info("모델 예열 중...")
        _ = pipe("모델 예열용 텍스트")
        logger.info("모델 예열 완료")

        return pipe

    except Exception as e:
        raise RuntimeError(f"모델 로딩 중 오류가 발생했습니다: {str(e)}") from e


# 모듈 레벨에서 모델 초기화
model = _load_model()


# 모델 인스턴스에 접근하기 위한 간단한 함수
def get_model():
    """
    초기화된 SBERT 모델 인스턴스 반환

    Returns:
        SentenceTransformer: 초기화된 한국어 SBERT 모델 인스턴스
    """
    return model
