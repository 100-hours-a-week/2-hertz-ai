import asyncio
import subprocess
import time

import GPUtil
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # MB 단위
    return 0


async def test_ollama():
    # 모든 모델 언로드
    subprocess.run(["ollama", "stop", "qwen2.5:7b"], capture_output=True)
    time.sleep(2)  # 언로드 대기

    # 모델 초기화
    ollama_llm = ChatOllama(model="qwen2.5:7b")

    # 간단한 에이전트 생성(도구 없음)
    agent = create_react_agent(ollama_llm, [])

    # # 테스트 메시지1
    # messages = [
    #     {"role": "user", "content": "55x55는?"},
    # ]
    # 테스트 메시지2
    messages = [
        {
            "role": "system",
            "content": "당신은 전문 기자입니다. 정확하고 객관적인 뉴스 기사를 작성해주세요. 답변은 json 형식으로 아래처럼 구성하세요:\n\n"
            '{ "title": "뉴스 제목", "content": "뉴스 내용" }',
        },
        {
            "role": "user",
            "content": "AI 기술 발전에 대한 짧은 뉴스 기사를 무조건 한국어로 작성해주세요.",
        },
    ]

    # 모델 실행 전 VRAM
    vram_before = get_vram_usage()

    # 모델 실행 및 시간 측정
    start_time = time.time()
    model_response = await agent.ainvoke({"messages": messages})
    end_time = time.time()

    # 모델 실행 전 VRAM
    vram_after = get_vram_usage()

    # 모델 응답 확인
    ai_message = model_response["messages"][-1]
    print(f"\n모델 응답:\n{ai_message}")

    # 응답시간 확인
    response_time = end_time - start_time
    print(f"\n응답시간: {response_time:.3f}초")

    # 토큰 확인
    token_usage = model_response["messages"][-1].usage_metadata
    token_per_second = token_usage["output_tokens"] / response_time
    print(f"\n토큰 수:\n{token_usage}")
    print(f"초당 처리 토큰 수:\n{token_per_second}")

    # VRAM 사용량 확인
    vram_used = vram_after - vram_before
    print(f"\nVRAM 사용량:\n {vram_used}MB")


if __name__ == "__main__":
    asyncio.run(test_ollama())
