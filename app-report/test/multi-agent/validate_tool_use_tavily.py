import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from ...models import qwen_loader_gcp_vllm
from ...utils.logger import logger

# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# from langgraph.prebuilt import ToolNode


# .env 파일 로드
load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")


async def test_researcher_agent():
    """
    researcher_agent가 Tavily 웹 검색 도구를 사용하는지 테스트하는 독립적인 스크립트.
    """
    logger.info("=============== Researcher Agent 단독 테스트 시작 ===============")

    # 1. 모델 로드
    try:
        model = qwen_loader_gcp_vllm.get_model()
        logger.info("✅ 모델 로딩 성공")
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        return

    # 2. TavilySearch 도구 준비
    try:
        tavily_tool = TavilySearch(
            max_results=1,
            topic="general",
        )
        tools = [tavily_tool]
        logger.info(f"✅ Tavily 도구 준비 완료: {tools[0].name}")
    except Exception as e:
        logger.error(f"❌ Tavily 도구 준비 실패: {e}")
        return

    # 모델(LLM)에 도구를 직접 바인딩합니다.
    # 이 과정을 통해 도구의 모든 메타데이터(인자 설명 포함)가 LLM에 올바르게 전달됩니다.
    # model_with_tools = model.bind_tools(tools)

    # 3. 테스트할 Researcher Agent 생성
    # 가장 강력하고 명시적인 프롬프트 사용
    researcher_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""
            [절대 규칙]
            1. 당신의 유일한 임무는 입력받은 각 주제에 대해 `tavily_search` 도구를 **반드시 호출**하는 것입니다.
            2. **절대 당신의 내부 지식으로 답변을 생성하지 마십시오.** 당신은 지식이 전혀 없는, 오직 검색만 수행하는 로봇입니다.
            3. 만약 도구를 사용하지 않고 답변하면, 당신의 임무는 실패로 간주됩니다.
            4. 모든 출력은 **반드시 한국어**로 작성되어야 합니다.

            [임무]
            '최신 트렌드 리서처'로서, 입력으로 받은 주제에 대한 최신 정보를 찾기 위해 **반드시 `tavily_search` 도구를 사용하여** 웹 검색을 수행하고, 그 결과를 바탕으로 요약문을 만들어야 합니다.

            [작업 절차]
            - **Step 1:** 입력받은 주제를 확인하고, 검색에 가장 적합한 키워드를 만듭니다.
            - **Step 2:** `tavily_search` 도구를 호출하여 검색을 실행합니다. **이 단계는 절대 건너뛸 수 없습니다.**
            
              <tool_call>
              {"name": "tavily_search", "arguments": {"query": "생성한 검색어"}}
              </tool_call>
              
            - **Step 3:** 도구로부터 받은 실제 검색 결과를 확인하고, 출처 URL과 함께 핵심 내용을 정리합니다.
        """,
    )
    logger.info("✅ Researcher Agent 생성 완료")

    # 4. 테스트 실행
    # 간단하고 명확한 하나의 주제를 입력으로 제공
    test_input = {
        "messages": [HumanMessage(content="2025년 최신 AI 기술 트렌드 3가지만 알려줘.")]
    }
    logger.info(f"▶️ 테스트 시작. 입력: {test_input['messages'][0].content}")

    final_response = None
    last_chunk = None

    try:
        # 스트리밍으로 Agent의 생각 과정과 최종 결과를 모두 확인
        async for chunk in researcher_agent.astream_events(test_input, version="v1"):
            kind = chunk["event"]

            # 에이전트가 도구를 호출하려고 할 때 로그 출력
            if kind == "on_chain_start" and chunk.get("name") == "agent":
                logger.info("--- Agent 실행 시작 ---")

            if kind == "on_tool_start":
                logger.info(
                    f"🛠️ 도구 호출 시작: {chunk['name']} | 입력: {chunk['data'].get('input')}"
                )

            # 에이전트의 중간 생각(rationale)을 확인
            if kind == "on_chat_model_stream":
                content = chunk["data"]["chunk"].content
                if content:
                    # 중간 생각 과정을 스트리밍으로 보여줌
                    print(content, end="", flush=True)

            # 도구 호출이 끝났을 때 로그 출력
            if kind == "on_tool_end":
                logger.info(
                    f"✅ 도구 호출 완료: {chunk['name']} | 결과: {str(chunk['data'].get('output'))[:200]}..."
                )

            # 에이전트 실행이 끝났을 때 최종 결과 저장
            if kind == "on_chain_end" and chunk.get("name") == "agent":
                logger.info("\n--- Agent 실행 종료 ---")
                final_response = chunk["data"].get("output")

        # 최종 결과 출력
        logger.info("\n\n=============== 최종 결과물 ===============")
        if final_response and final_response.get("messages"):
            # messages 리스트의 마지막 메시지가 최종 답변
            final_message = final_response["messages"][-1]
            logger.info(f"최종 생성된 콘텐츠:\n{final_message.content}")

            # tool_calls가 있었는지 확인
            if any(msg.tool_calls for msg in final_response["messages"]):
                logger.info("🎉 성공: 에이전트가 도구를 성공적으로 사용했습니다.")
            else:
                logger.warning(
                    "⚠️ 실패: 에이전트가 도구를 사용하지 않고 답변을 생성했습니다."
                )
        else:
            logger.error("❌ 최종 결과물을 찾을 수 없습니다.")

    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    # 스크립트 실행
    # 프로젝트 루트 디렉토리에서 다음 명령어로 실행하세요:
    # python -m app_report.test.multi_agent.test_researcher
    asyncio.run(test_researcher_agent())
