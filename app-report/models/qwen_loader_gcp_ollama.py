from dotenv import load_dotenv
from langchain_ollama import ChatOllama


# 모듈 임포트 시점에 바로 모델 초기화
def _load_model():

    load_dotenv()

    loaded_model = ChatOllama(model="qwen2.5:7b")

    # 모델 예열 (첫 추론 시간 단축)
    # _ = loaded_model.invoke("모델 예열용 텍스트", max_token=50, temperature=0.91)

    return loaded_model


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
