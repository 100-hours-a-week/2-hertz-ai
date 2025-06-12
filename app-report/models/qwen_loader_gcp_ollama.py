import subprocess
import time

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from ..utils.logger import log_performance, logger


def get_gpu_memory_usage(device_index=0) -> float:
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        mems = result.decode().strip().split("\n")
        return float(mems[device_index])
    except Exception as e:
        logger.warning(f"nvidia-smi 실행 실패: {e}")
        return -1.0


# 모듈 임포트 시점에 바로 모델 초기화
@log_performance(operation_name="load_ollama_model", include_memory=True)
def _load_model():
    """
    Ollama Qwen2.5:7b 모델을 로딩합니다.
    """

    load_dotenv()
    try:
        logger.info(" Ollama[Qwen2.5:7b] 모델 로딩 시작...")

        start_time = time.time()
        gpu_mem_before = get_gpu_memory_usage()

        # 모델 로드
        loaded_model = ChatOllama(model="qwen2.5:7b")
        # 모델 예열 (첫 추론 시간 단축)
        # _ = loaded_model.invoke("모델 예열용 텍스트", max_token=50, temperature=0.91)

        elapsed = round(time.time() - start_time, 2)
        gpu_mem_after = get_gpu_memory_usage()
        gpu_diff = gpu_mem_after - gpu_mem_before

        logger.info(
            f" Ollama 모델 로딩 완료: {elapsed}s | GPU 사용량: {gpu_mem_after:.2f}MB (+{gpu_diff:.2f}MB)"
        )

        return loaded_model

    except Exception as e:
        logger.exception(f" 모델 로딩 실패: {e}")
        raise


# 모듈 레벨에서 모델 초기화
model = _load_model()


# 모델 인스턴스에 접근하기 위한 간단한 함수
def get_model():
    """
    초기화된 Qwen 모델 인스턴스 반환

    Returns:
        Qwen: 초기화된 Qwen 모델 인스턴스
    """
    return model
