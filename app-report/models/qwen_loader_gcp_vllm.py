import time

import GPUtil

# from dotenv import load_dotenv
# from langchain_community.llms import VLLM
from langchain_openai import ChatOpenAI

from ..utils.logger import log_performance, logger


def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # MB 단위
    return 0


# 모듈 임포트 시점에 바로 모델 초기화
@log_performance(operation_name="load_vllm_model", include_memory=True)
def _load_model():
    """
    VLLM Qwen2-7B-Instruct 모델을 로딩합니다.
    """

    try:
        logger.info("모델 로딩 시작...")

        start_time = time.time()
        gpu_mem_before = get_vram_usage()
        # 모델 로드
        inference_server_url = "http://localhost:8001/v1"
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        loaded_model = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
        )
        logger.info(f"VLLM[{model_name}] 모델 로딩 완료!")
        # 모델 예열 (첫 추론 시간 단축)
        # _ = loaded_model.invoke("모델 예열용 텍스트", max_token=50, temperature=0.91)

        elapsed = round(time.time() - start_time, 2)
        gpu_mem_after = get_vram_usage()
        gpu_diff = gpu_mem_after - gpu_mem_before

        logger.info(
            f" VLLM 모델 로딩 완료: {elapsed}s | GPU 사용량: {gpu_mem_after:.2f}MB (+{gpu_diff:.2f}MB)"
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
