import time

import GPUtil
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # MB 단위
    return 0


# 모듈 임포트 시점에 바로 모델 초기화
# @log_performance(operation_name="load_vllm_model", include_memory=True)
def _load_model():
    """
    VLLM Qwen2-7B-Instruct 모델을 로딩합니다.
    """

    try:
        # logger.info(" VLLM[Qwen2-7B-Instruct] 모델 로딩 시작...")

        start_time = time.time()
        gpu_mem_before = get_vram_usage()
        # 모델 로드
        inference_server_url = "http://localhost:8001/v1"
        loaded_model = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
        )
        # 모델 예열 (첫 추론 시간 단축)
        # _ = loaded_model.invoke("모델 예열용 텍스트", max_token=50, temperature=0.91)

        elapsed = round(time.time() - start_time, 2)
        gpu_mem_after = get_vram_usage()
        gpu_diff = gpu_mem_after - gpu_mem_before

        # logger.info(
        #     f" VLLM 모델 로딩 완료: {elapsed}s | GPU 사용량: {gpu_mem_after:.2f}MB (+{gpu_diff:.2f}MB)"
        # )
        return loaded_model

    except Exception as e:
        # logger.exception(f" 모델 로딩 실패: {e}")
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


def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


model = get_model()

flight_assistant = create_react_agent(
    model=model,
    tools=[book_flight],
    prompt="You are a flight booking assistant. When you receive flight booking requests, immediately book the flight using the book_flight tool with the provided information. Do not ask for confirmation - just execute the booking.",
    name="flight_assistant",
)

hotel_assistant = create_react_agent(
    model=model,
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant",
)

# model : Qwen/Qwen2.5-7B-Instruct-AWQ
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=model,
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    ),
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK on January 15th and a stay at McKittrick Hotel from January 15th to January 18th",
            }
        ]
    }
):
    print(chunk)
    print("\n")

input_data = {
    "messages": [
        {
            "role": "user",
            "content": "book a flight from BOS to JFK on January 15th and a stay at McKittrick Hotel from January 15th to January 18th",
        }
    ]
}


def extract_final_supervisor_response(supervisor, input_data):
    """마지막 supervisor 응답만 추출하여 출력"""
    all_chunks = []

    # 모든 chunk를 수집
    for chunk in supervisor.stream(input_data):
        if chunk:
            all_chunks.append(chunk)

    # 마지막 supervisor 응답 찾기
    for chunk in reversed(all_chunks):
        if "supervisor" in chunk and chunk["supervisor"]:
            if "messages" in chunk["supervisor"]:
                messages = chunk["supervisor"]["messages"]
                # 마지막 AI 메시지 찾기
                for message in reversed(messages):
                    if hasattr(message, "content") and hasattr(message, "name"):
                        if message.name == "supervisor" and message.content.strip():
                            print("🎯 최종 예약 결과:")
                            print("=" * 50)
                            print(message.content)
                            print("=" * 50)

    print("❌ Supervisor의 최종 응답을 찾을 수 없습니다.")


extract_final_supervisor_response(supervisor, input_data)
