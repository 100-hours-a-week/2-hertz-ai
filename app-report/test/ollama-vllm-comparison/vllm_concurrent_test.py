import asyncio
import statistics
import subprocess
import time

import GPUtil
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # MB 단위
    return 0


async def test_vllm():
    # 모든 모델 언로드
    # subprocess.run(["ollama", "stop", "qwen2.5:7b"], capture_output=True)
    time.sleep(2)  # 언로드 대기

    # 모델 초기화
    inference_server_url = "http://localhost:8000/v1"
    vllm_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
    )

    # 간단한 에이전트 생성(도구 없음)
    agent = create_react_agent(vllm_llm, [])

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
            "content": "AI 기술 발전에 대한 짧은 뉴스 기사를 한국어로 작성해주세요.",
        },
    ]

    # 모델 실행 전 VRAM
    vram_before = get_vram_usage()

    # 모델 실행 및 시간 측정
    start_time = time.time()
    model_response = await agent.ainvoke({"messages": messages})
    end_time = time.time()

    print(f"\n모델 응답:\n{model_response}")

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

    # VRAM 사용량 확인(vLLM 서버를 먼저 실행하기에 측정 불가)
    vram_used = vram_after - vram_before
    print(f"\nVRAM 사용량:\n {vram_used}MB")


async def concurrent_request(llm, agent, request_id):
    """단일 동시 요청 처리"""
    messages = [
        {
            "role": "user",
            "content": f"요청 {request_id}: {request_id * 10 + 5}의 제곱과 제곱근을 구해주세요.",
        }
    ]

    try:
        start_time = time.time()
        response = await agent.ainvoke({"messages": messages})
        end_time = time.time()

        ai_message = response["messages"][-1]
        response_time = end_time - start_time
        token_usage = ai_message.usage_metadata

        print(f"\n모델 응답:\n{ai_message}")

        return {
            "request_id": request_id,
            "success": True,
            "response_time": response_time,
            "tokens_per_second": token_usage["output_tokens"] / response_time,
            "output_tokens": token_usage["output_tokens"],
        }
    except Exception as e:
        return {"request_id": request_id, "success": False, "error": str(e)}


async def test_concurrent_ollama(num_requests=3):
    """동시 요청 테스트 - 기존 코드에 추가하기 좋은 간단한 버전"""
    print(f"\n{'='*50}")
    print(f"🧪 동시 요청 테스트 ({num_requests}개)")
    print(f"{'='*50}")

    # 모든 모델 언로드
    subprocess.run(["ollama", "stop", "qwen2.5:7b"], capture_output=True)
    time.sleep(2)

    # 메모리 측정 시작
    vram_before = get_vram_usage()

    # 모델 초기화
    inference_server_url = "http://localhost:8000/v1"
    vllm_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
    )
    agent = create_react_agent(vllm_llm, [])

    # 동시 요청 실행
    print(f"⏱️  {num_requests}개 동시 요청 시작...")
    overall_start = time.time()

    tasks = [concurrent_request(vllm_llm, agent, i + 1) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    overall_end = time.time()
    overall_time = overall_end - overall_start

    # 메모리 측정 종료
    vram_after = get_vram_usage()

    # 결과 분석
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    print(f"\n📊 결과:")
    print(f"   ✅ 성공: {len(successful)}/{num_requests}")
    print(f"   ❌ 실패: {len(failed)}")
    print(f"   ⏱️  전체 시간: {overall_time:.2f}초")

    if successful:
        response_times = [r["response_time"] for r in successful]
        tokens_per_sec = [r["tokens_per_second"] for r in successful]

        print(f"   📈 평균 응답시간: {statistics.mean(response_times):.2f}초")
        print(f"   🚀 평균 토큰/초: {statistics.mean(tokens_per_sec):.1f}")
        print(f"   📊 총 토큰: {sum(r['output_tokens'] for r in successful)}")

    print(f"   💾 VRAM 사용: {vram_after - vram_before}MB")

    # 개별 결과
    print(f"\n📋 개별 결과:")
    for result in successful:
        print(f"   요청 {result['request_id']}: {result['response_time']:.2f}초")

    for result in failed:
        print(f"   ❌ 요청 {result['request_id']}: {result['error']}")


async def main():
    """메인 실행 함수"""
    print("Ollama 테스트를 시작합니다!")
    print("\n1. 단일 요청 테스트")
    print("2. 동시 요청 테스트 (3개)")
    print("3. 동시 요청 테스트 (5개)")
    print("4. 모든 테스트 실행")

    choice = input("\n선택하세요 (1-4): ").strip()

    if choice == "1":
        await test_vllm()
    elif choice == "2":
        await test_concurrent_ollama(3)
    elif choice == "3":
        await test_concurrent_ollama(5)
    elif choice == "4":
        print("\n🔄 모든 테스트를 순차 실행합니다...")

        # 단일 요청 테스트
        await test_vllm()

        # 3초 대기
        print("\n⏳ 3초 후 동시 요청 테스트...")
        await asyncio.sleep(3)

        # 3개 동시 요청
        await test_concurrent_ollama(3)

        # 3초 대기
        print("\n⏳ 3초 후 5개 동시 요청 테스트...")
        await asyncio.sleep(3)

        # 5개 동시 요청
        await test_concurrent_ollama(5)

        print("\n🎉 모든 테스트 완료!")
    else:
        print("잘못된 선택입니다. 기본으로 단일 테스트를 실행합니다.")
        await test_vllm()


if __name__ == "__main__":
    asyncio.run(main())
