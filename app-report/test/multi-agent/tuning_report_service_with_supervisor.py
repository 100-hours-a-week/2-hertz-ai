import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from ...models import qwen_loader_gcp_vllm
from ...utils.logger import logger

# from .async_tool_wrapper import AsyncToolWrapper
from .tavily_search import TavilySearch

load_dotenv()

# def load_mcp_config():
#     """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
#     try:
#         # parent_dir, _ = os.path.split(os.path.dirname(__file__))
#         # config_path = os.path.join(parent_dir, "config", "mcp_config.json")
#         current_file_path = os.path.abspath(__file__)
#         app_report_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
#         config_path = os.path.join(app_report_dir, "config", "mcp_config.json")
#         with open(config_path, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         logger.error(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
#         return {}

# def create_server_config():
#     """MCP 서버 설정을 생성합니다."""
#     config = load_mcp_config()
#     server_config = {}

#     if config and "mcpServers" in config:
#         for server_name, server_config_data in config["mcpServers"].items():
#             # command가 있으면 stdio 방식
#             if "command" in server_config_data:
#                 server_config[server_name] = {
#                     "command": server_config_data.get("command"),
#                     "args": server_config_data.get("args", []),
#                     "transport": "stdio",
#                 }
#             # url이 있으면 sse 방식
#             elif "url" in server_config_data:
#                 server_config[server_name] = {
#                     "url": server_config_data.get("url"),
#                     "transport": "sse",
#                 }
#     if not server_config:
#         logger.warning("⚠ MCP 서버 설정이 비어 있습니다. MCP 기능이 비활성화됩니다.")

#     return server_config


async def main():
    # model - Qwen/Qwen2.5-7B-Instruct-AWQ
    model = qwen_loader_gcp_vllm.get_model()

    # 관계 분석 에이전트
    connection_finder_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""
            [절대 규칙]
            - 당신의 유일한 출력 언어는 '한국어'입니다. 다른 어떤 언어도 사용해서는 안 됩니다.
            - 절대 입력된 프로필 정보 외의 내용을 추측하거나 만들어내지 마십시오.
            - 다른 인사나 부가 설명 없이, 분석 요약문만 즉시 출력하십시오.

            [임무]
            당신은 고도로 훈련된 '관계 분석 전문가'입니다. 입력으로 주어진 두 사용자의 프로필 데이터를 바탕으로, 두 사람의 **공통점**과 **상호보완적인 특징**을 찾아내어 한 문단의 간결한 텍스트로 요약해야 합니다.

            [출력 예시]
            김지훈님은 내향적(INTP)이고 조용한 활동을 선호하는 반면, 박서연님은 외향적(ENFJ)이고 새로운 경험을 즐깁니다. 두 분은 음악 감상과 파스타라는 공통 관심사를 가지고 있어 이를 시작점으로 대화를 풀어나가기 좋으며, 서로 다른 성향과 데이트 취향은 각자에게 새로운 영감과 활력을 불어넣어 줄 수 있는 훌륭한 상호보완적 관계입니다.
        """,
        name="connection_finder_assistant",
    )

    # 주제 기획 에이전트
    topic_planner_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""
            당신은 창의적인 '콘텐츠 기획자'입니다.

            당신의 임무는 이전 단계에서 전달받은 '관계 분석 요약문'을 읽고, 그 내용을 기반으로 흥미로운 콘텐츠 주제 3개를 제안하는 것입니다.

            **규칙:**
            - 제안하는 주제는 반드시 분석된 내용(공통점, 특징)과 관련 있어야 합니다.
            - 출력은 **반드시** 아래와 같은 Python 리스트 형식의 문자열이어야 합니다.
            - 예시 형식: `['두 사람의 MBTI 궁합 파헤치기', '공통 관심사인 영화 취향으로 본 데이트 코스', '서로 다른 입맛을 모두 만족시킬 맛집 탐방 코스']`
            - 다른 어떤 텍스트도 추가하지 말고, 오직 Python 리스트 형식의 문자열만 출력하십시오.
        """,
        name="topic_planner_assistant",
    )

    # 정보 검색 에이전트
    # server_config = create_server_config()
    # client = MultiServerMCPClient(server_config)
    # try:
    #     mcp_tools = await client.get_tools()
    #     logger.info(f"✅ MCP 도구 로드 성공: {len(mcp_tools)}개")

    #     research_tools = [AsyncToolWrapper(tool) for tool in mcp_tools]

    # except Exception as e:
    #     logger.info(f"⚠️ MCP 도구 로드 실패: {e}")
    #     mcp_tools = []
    research_tools = [TavilySearch(max_results=5)]

    researcher_agent = create_react_agent(
        model=model,
        tools=research_tools,
        prompt="""
            [절대 규칙]
            - 당신의 유일한 출력 언어는 '한국어'입니다. 다른 어떤 언어도 사용해서는 안 됩니다.
            - 당신은 내부에 저장된 지식이 전혀 없는, 오직 `tavily_web_search` 도구만 사용할 수 있는 검색 로봇입니다. **절대 당신의 지식으로 답변을 창작하지 마십시오.**
            - 각 주제에 대해 **반드시, 그리고 개별적으로** `tavily_web_search` 도구를 호출해야 합니다.
            - 도구 호출 후 받은 결과는 JSON 형식의 문자열입니다. 이 문자열 안의 각 검색 결과(`title`, `url`, `content`)를 바탕으로, 다음 작가가 사용할 수 있도록 **출처(URL)와 함께 핵심 내용만** 간결한 불렛 포인트(`-`)로 요약하여 정리하십시오.

            [임무]
            당신은 '최신 트렌드 리서처'입니다. 이전 단계에서 받은 콘텐츠 주제 리스트의 각 항목에 대해, **2025년 최신 트렌드와 정보를** 웹에서 검색하여 다음 작가 에이전트가 사용할 수 있도록 핵심 내용을 요약해야 합니다.

            [작업 순서 (매우 중요)]
            1.  입력받은 주제 리스트(`['주제1', '주제2', ...]`)를 확인합니다.
            2.  **첫 번째 주제**에 대한 최신 정보를 얻기 위한 검색어(예: "INTP ENFJ 데이트 코스 2024년 최신")를 만듭니다.
            3.  `tavily_web_search` 도구를 해당 검색어로 호출합니다.
            4.  도구로부터 받은 검색 결과를 바탕으로, **출처(URL)를 포함하여** 핵심 내용을 불렛 포인트(-)로 요약합니다.
            5.  **다음 주제**에 대해 2~4번 과정을 반복합니다.
            6.  모든 주제에 대한 검색 및 요약이 끝나면 작업을 종료합니다.

            [출력 형식 예시]
            **주제: INTP와 ENFJ의 MBTI 궁합 분석**
            - INTP와 ENFJ는 '멘토-제자' 관계로 불리며, ENFJ가 INTP의 잠재력을 이끌어내는 상호보완적 관계라는 분석이 많음 (출처: https://...)
            - 2024년 커뮤니티 트렌드에 따르면, 이 조합은 깊은 철학적 대화에서 큰 만족감을 얻는다는 후기가 다수 발견됨 (출처: https://...)

            **주제: 조용한 데이트와 맛집 탐방을 결합한 코스**
            - 최근 '북카페'와 '보드게임 카페'가 결합된 복합 문화 공간이 인기를 끌고 있음. 조용한 대화와 활동적인 재미를 동시에 만족시킬 수 있음 (출처: https://...)
        """,
        name="researcher_assistant",
    )

    # 창의적 컨셉 에이전트
    creative_concept_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""
            당신은 트렌드에 민감한 '크리에이티브 디렉터'이자 재치 있는 '카피라이터'입니다.

            당신의 임무는 실제 장문의 글을 쓰는 것이 아니라, 입력된 정보를 바탕으로 최종 콘텐츠의 **핵심 뼈대가 될 '콘텐츠 기획안'**을 만드는 것입니다.

            **입력 정보:**
            - 관계 분석 요약
            - 콘텐츠 주제 리스트
            - 관련 검색 결과

            **기획안 작성 가이드:**
            1.  **컨셉 결정:** 제공된 정보를 보고, 전체 콘텐츠를 어떤 느낌으로 만들지 결정하세요. (예: SNS 밈 스타일, 커뮤니티 폭로 글, 짧은 시트콤 시나리오, 진지한 분석 리포트 등) **뉴스 형식에 얽매일 필요 전혀 없습니다.**
            2.  **헤드라인 제안:** 결정된 컨셉에 맞는, 사람들의 시선을 확 끄는 헤드라인 아이디어를 1~2개 제안하세요.
            3.  **섹션별 핵심 메시지 설계:** 글의 각 부분에 어떤 내용이 들어갈지, 핵심 키워드나 짧은 문장(앵커 텍스트)으로 설계해주세요. 이모지를 활용하여 톤앤매너를 표현할 수 있습니다.

            **규칙:**
            - 절대 긴 문장으로 본문을 작성하지 마세요. 당신의 역할은 오직 '기획'입니다.
            - 결과물은 아래 예시와 같이 구조화된 텍스트 형식이어야 합니다.

            ---
            **기획안 출력 예시:**

            **컨셉:** 찐친 바이브 폭발하는 SNS 밈 스타일 게시물

            **헤드라인 아이디어:**
            - "🚨[우정경보] ESTP x ENFP, 이 조합 지구에 무슨 짓을 한거야?!"
            - "속보) 대화 424번, 걔네 그냥 친구 아니래"

            **섹션별 핵심 메시지:**
            - **오프닝:** "이건 그냥 친구가 아냐. 우주급 사건이다." 라는 문구와 함께 대화 횟수 강조. 빅뱅 이모지 💥 사용.
            - **MBTI 케미:** "롤러코스터", "사건 제조기" 키워드로 둘의 시너지 표현. 표(테이블) 형식으로 특징 비교.
            - **취미 분석:** "취미도 어딘가 정상 아님" 이라는 소제목. 각 취미를 한 줄 밈으로 요약. (예: 브런치? -> 단순 식사 아님, 예술임)
            - **미래 예측:** "이 우정, 넷플릭스 입장하세요" 라는 컨셉으로 가상 드라마 정보 생성.
            - **마무리:** "쿠키 영상 있음" 이라는 소제목으로 다음 만남에 대한 기대감 증폭. #도파민_대폭발 같은 해시태그 사용.
            ---

            이제, 주어진 정보를 바탕으로 위 가이드에 맞춰 창의적인 '콘텐츠 기획안'을 작성하세요.
        """,
        name="creative_concept_assistant",
    )

    # 컨텐츠 생성 에이전트
    content_generator_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""
            당신은 뛰어난 필력을 가진 '콘텐츠 작가'입니다.
            **모든 출력은 반드시 한국어로 작성해야 합니다.**

            당신의 유일한 임무는, 이전 단계의 크리에이티브 디렉터가 작성한 **'콘텐츠 기획안'을 입력으로 받아, 그 뼈대에 살을 붙여 재미있고 완성도 높은 최종 글을 작성**하는 것입니다.

            **매우 중요한 규칙:**
            - **절대** '콘텐츠 기획안'을 그대로 복사해서는 안 됩니다.
            - 기획안에 제시된 **컨셉, 톤앤매너, 헤드라인, 섹션별 핵심 메시지**를 충실히 따르면서, 각 섹션을 **자연스러운 문장과 풍부한 묘사가 담긴 완전한 문단**으로 만들어야 합니다.
            - 당신의 목표는 뼈대(기획안)를 바탕으로 멋진 최종 글을 **창조**하는 것입니다. 이제 주어진 '콘텐츠 기획안'을 바탕으로 최종 글을 작성하세요.
        """,
        name="content_generator_assistant",
    )

    # 검수 및 편집 에이전트
    editor_agent = create_react_agent(
        model=model,
        tools=[],
        prompt="""
            당신은 오직 하나의 임무만 수행하는 '데이터 포맷터'입니다.
            **모든 출력은 반드시 한국어로 작성해야 합니다.**

            당신의 유일한 임무는 입력받은 '제목'과 '본문' 텍스트를 **단 하나의 유효한 JSON 객체**로 만드는 것입니다.

            **절대 규칙:**
            1.  입력 텍스트를 절대 수정하지 마십시오.
            2.  당신의 최종 출력물은 **오직 JSON 객체 하나**여야 합니다. JSON 앞뒤로 어떤 글자나 설명도 추가하지 마십시오.
            3.  출력 형식: `{"title": "입력받은 제목", "content": "입력받은 본문 내용"}`
            4.  본문 내용(`content`)의 모든 줄바꿈은 `\\n`으로 이스케이프 처리해야 합니다.
        """,
        name="editor_assistant",
    )

    # 감독 에이전트
    supervisor = create_supervisor(
        agents=[
            connection_finder_agent,
            topic_planner_agent,
            researcher_agent,
            creative_concept_agent,
            content_generator_agent,
            editor_agent,
        ],
        model=model,
        prompt=(
            """
            당신은 고도로 체계적인 'AI 워크플로우 매니저'입니다. 당신의 임무는 각 전문가 에이전트에게 작업을 순서대로 위임하고, 한 단계의 결과물을 다음 단계의 입력으로 정확하게 전달하여 최종 보고서를 완성하는 것입니다.

            **매우 중요한 규칙:**
            - 당신은 반드시 아래에 정의된 6단계 작업 흐름을 따라야 합니다.
            - 절대 단계를 건너뛰거나 임의로 작업을 종료해서는 안 됩니다.
            - 각 단계를 실행하기 전에, 먼저 "나는 지금 몇 단계를 수행해야 하는가? 다음으로 호출할 에이전트는 누구인가?"를 속으로 생각한 후, 해당 에이전트를 호출하는 도구를 사용하십시오.

            **작업 흐름:**
            1.  **[1단계: 관계 분석]** -> `connection_finder_assistant` 호출
            2.  **[2단계: 주제 기획]** -> `topic_planner_assistant` 호출
            3.  **[3단계: 정보 검색]** -> `researcher_assistant` 호출
            4.  **[4단계: 컨셉 설계]** -> `creative_concept_assistant` 호출
            5.  **[5단계: 본문 생성]** -> `content_generator_assistant` 호출
            6.  **[6단계: 최종 포맷팅]** -> `editor_assistant` 호출

            **최종 행동 지침:**
            `editor_assistant`로부터 최종 JSON을 받으면, 그것이 이 임무의 완전한 끝입니다. 더 이상 다른 에이전트를 호출하지 말고, 받은 JSON을 그대로 최종 답변으로 출력하며 작업을 종료하십시오.
            """
        ),
    ).compile()

    final_response_content = None
    last_chunk = None

    async for chunk in supervisor.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        """
                        두 사용자를 위한 흥미로운 매칭 리포트를 작성해주세요. 전체 작업은 AI 워크플로우 매니저의 지시에 따라 자동으로 진행될 것입니다. 최종 결과물만 생성해주시면 됩니다.

                        ---
                        **사용자 A 프로필:**
                        - 이름: 김지훈
                        - MBTI: INTP
                        - 관심사: 음악 감상, 넷플릭스 보기, 코딩, 보드게임
                        - 좋아하는 음식: 파스타, 피자
                        - 선호하는 데이트: 조용한 카페에서 대화하기, 집에서 함께 영화보기

                        **사용자 B 프로필:**
                        - 이름: 박서연
                        - MBTI: ENFJ
                        - 관심사: 음악 감상, 맛집 탐방, 봉사활동, 사람들과 어울리기
                        - 좋아하는 음식: 파스타, 떡볶이
                        - 선호하는 데이트: 새로운 장소 탐방하기, 함께 맛있는 음식 먹기
                        ---
                        """
                    ),
                }
            ]
        }
    ):
        logger.debug(f"--- 스트리밍 청크 ---\n{chunk}")
        # 모든 청크를 순회하며 마지막 청크를 계속 덮어씁니다.
        last_chunk = chunk

    logger.info("\n\n=============== 최종 결과물 ===============")

    # 루프가 끝난 후, 마지막 청크를 분석합니다.
    if last_chunk:
        # 마지막 청크의 키가 'supervisor'이고, 그 안에 메시지가 있는지 확인
        if "supervisor" in last_chunk and last_chunk["supervisor"].get("messages"):
            # supervisor의 메시지 리스트에서 마지막 메시지를 가져옵니다.
            last_message = last_chunk["supervisor"]["messages"][-1]

            # 마지막 메시지가 AI의 응답이고, tool_calls가 없는 경우 최종 답변으로 간주합니다.
            if last_message.tool_calls is None or len(last_message.tool_calls) == 0:
                final_response_content = last_message.content
                logger.info(f"최종 생성된 콘텐츠:\n{final_response_content}")
            else:
                logger.error(
                    "마지막 청크가 Tool Call을 포함하고 있어 최종 결과물이 아닙니다."
                )
        else:
            logger.error("마지막 청크에 supervisor의 메시지가 없습니다.")
    else:
        logger.error("최종 결과물을 찾을 수 없습니다. (스트림이 비어 있음)")


if __name__ == "__main__":
    asyncio.run(main())
