import time

import GPUtil
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..utils.logger import log_performance, logger


def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # MB 단위
    return 0


# 모듈 임포트 시점에 바로 모델 초기화
@log_performance(operation_name="load_midm_model", include_memory=True)
def _load_model_and_tokenizer():
    """
    Midm-2.0-Base-Instruct 모델을 로딩합니다.
    """

    try:
        logger.info("모델 로딩 시작...")

        start_time = time.time()
        gpu_mem_before = get_vram_usage()

        model_name = "K-intelligence/Midm-2.0-Base-Instruct"

        # 토크나이저 로드
        logger.info("토크나이저 로딩 중...")
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # 모델 로드
        logger.info("모델 로딩 중...")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
        )
        logger.info(f"{model_name} 모델 로딩 완료!")

        # 모델 예열 (첫 추론 시간 단축)
        # _ = loaded_model.invoke("모델 예열용 텍스트", max_token=50, temperature=0.91)

        elapsed = round(time.time() - start_time, 2)
        gpu_mem_after = get_vram_usage()
        gpu_diff = gpu_mem_after - gpu_mem_before

        logger.info(
            f" VLLM 모델 로딩 완료: {elapsed}s | GPU 사용량: {gpu_mem_after:.2f}MB (+{gpu_diff:.2f}MB)"
        )
        return loaded_model, loaded_tokenizer

    except Exception as e:
        logger.exception(f" 모델 로딩 실패: {e}")
        raise


# 모듈 레벨에서 모델 초기화
_model, _tokenizer = _load_model_and_tokenizer()

pipe = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=1024,
    return_full_text=False,
)

hf_pipeline = HuggingFacePipeline(pipeline=pipe)

chat_model_wrapper = ChatHuggingFace(llm=hf_pipeline)


# 모델 인스턴스에 접근하기 위한 간단한 함수
def get_model():
    """
    초기화된 Midm 모델 인스턴스 반환

    Returns:
        Midm: 초기화된 모델 인스턴스
    """
    return model


def get_langchain_model():
    """
    초기화된 LangChain 호환 모델 인스턴스 반환
    """
    return chat_model_wrapper
