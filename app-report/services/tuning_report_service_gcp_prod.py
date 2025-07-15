import json
import random
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import create_react_agent

from ..core.enum_process import convert_to_korean
from ..models.midm_loader_gcp_vllm import get_model
from ..schemas.tuning_schema import TuningReport, TuningReportResponse
from ..utils.logger import log_performance, logger

load_dotenv()


def _format_profile_for_prompt(profile: dict, user_label: str) -> str:
    """
    사용자 프로필 딕셔너리를 프롬프트에 넣기 좋은 문자열 형식으로 변환합니다.
    """
    # convert_to_korean 함수를 사용하여 Enum 값을 한국어로 변환
    korean_profile = convert_to_korean(profile)

    # 각 항목을 줄바꿈하며 문자열로 조합
    lines = [f"**{user_label} 프로필:**"]
    for key, value in korean_profile.items():
        label_map = {
            "ageGroup": "나이대",
            "MBTI": "MBTI",
            "gender": "성별",
            "religion": "종교",
            "smoking": "흡연",
            "drinking": "음주",
            "personality": "성격",
            "preferredPeople": "선호하는사람",
            "currentInterests": "관심사",
            "favoriteFoods": "좋아하는 음식",
            "likedSports": "즐기는 운동",
            "pets": "애완동물",
            "selfDevelopment": "자기계발",
            "hobbies": "취미",
        }

        # 관심 있는 항목만 선택적으로 포함
        if key in label_map:
            label = label_map[key]
            # 값이 리스트인 경우, 쉼표로 구분된 문자열로 변환
            if isinstance(value, list):
                value_str = ", ".join(value)
            else:
                value_str = str(value)
            lines.append(f"- {label}: {value_str}")

    return "\n".join(lines)


# 대체 헤드라인 랜덤 선택
def get_fallback_title():
    """
    예기치 못한 에러 상황을 위한 범용적인 대체 제목을 랜덤으로 선택

    Returns:
        str: 랜덤하게 선택된 대체 제목
    """
    fallback_titles = [
        "🌟 오늘의 특별한 이야기가 준비되었습니다! 🌟",
        "💫 지금 바로 확인해야 할 흥미로운 소식! 💫",
        "🎯 놓치면 후회할 핫한 콘텐츠 대공개! 🎯",
        "✨ 화제의 주인공들이 전하는 특별한 메시지 ✨",
        "🚀 오늘 가장 주목받는 이야기를 만나보세요! 🚀",
        "🔥 지금 이 순간, 가장 HOT한 소식 전격 공개! 🔥",
        "🌈 일상을 바꿔줄 특별한 인사이트가 도착했어요! 🌈",
        "⭐ 모두가 궁금해하는 그 이야기, 드디어 공개! ⭐",
        "💎 오늘만 볼 수 있는 특급 콘텐츠를 확인하세요! 💎",
        "🎪 예상치 못한 재미가 가득한 이야기 시간! 🎪",
    ]

    return random.choice(fallback_titles)


# State 정의
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 각 단계별 결과를 저장하기 위한 필드들 추가
    connection_analysis: str  # 1단계: 관계 분석 결과
    topic_list: str  # 2단계: 주제 기획 결과
    research_results: str  # 3단계: 정보 검색 결과
    creative_concept: str  # 4단계: 컨셉 설계 결과
    title: str
    final_content: str  # 5단계: 본문 생성 결과
    json_output: str  # 6단계: 최종 포맷팅 결과


@log_performance(
    operation_name="generate_tuning_report_with_agent",
    include_memory=True,
    include_args=True,
)
def generate_tuning_report_with_agent(request: TuningReport) -> TuningReportResponse:
    """
    Multi-Agent LangGraph 워크플로우를 실행하여 튜닝 리포트를 생성하는 서비스 함수
    """
    try:

        # model - K-intelligence/Midm-2.0-Base-Instruct
        model = get_model()

        # 관계 분석 에이전트
        connection_finder_agent = create_react_agent(
            model=model,
            tools=[],
            prompt="""
                [절대 규칙]
                - 당신의 유일한 출력 언어는 '한국어'입니다. 다른 어떤 언어도 사용해서는 안 됩니다.
                - 절대 입력된 프로필 정보 외의 내용을 추측하거나 만들어내지 마십시오.
                - 다른 인사나 부가 설명 없이, 분석 요약문만 즉시 출력하십시오.

                당신은 두 사람의 관계에서 터질 '대박 사건'의 냄새를 기가 막히게 맡는 **'관계 가십 폭로 전문가'**입니다. 입력된 두 사용자의 프로필을 보고, 이 둘이 만났을 때 벌어질 **가장 흥미롭고 극적인 '케미 포인트' 딱 하나**를 찾아내어, 마치 특종을 터뜨리듯 한 문장으로 요약하세요.

                [출력 예시]
                - 계획의 화신 INTJ와 자유로운 영혼 ENFP가 만났다고? 이건 서로의 인생에 가장 큰 버그이자 최고의 업데이트가 될 운명적 만남이다!
                - 조용한 집순이와 핵인싸가 '음악'과 '영화'라는 같은 취미를 가졌다는 건, 서로의 세계를 통째로 뒤흔들 전쟁의 서막이나 다름없다.
            """,
            name="connection_finder_assistant",
        )

        # 주제 기획 에이전트
        topic_planner_agent = create_react_agent(
            model=model,
            tools=[],
            prompt="""
                당신은 자극적인 콘텐츠로 조회수를 뽑아내는 **'유튜브 바이럴 콘텐츠 PD'**입니다.

                이전 단계에서 받은 '케미 포인트'를 기반으로, 독자들이 클릭하지 않고는 못 배길 **완전히 다르고 예측 불가능한 3가지 관점의 '어그로성 소제목'**을 자유롭게 기획해야 합니다.

                **[콘텐츠 앵글 '아이디어 뱅크']**
                - **MBTI 심층 분석:** 단순한 충돌이 아닌, 특정 상황에서의 심리 분석 (예: "INTJ가 파티에서 살아남는 법")
                - **가상 시나리오:** "만약 두 사람이 같이 창업을 한다면?", "무인도에 둘만 남는다면?"
                - **데이터 탐정:** 두 사람의 프로필(취미, 관심사 등)을 데이터로 보고 숨겨진 의미를 파헤치기 (예: "두 사람의 넷플릭스 시청 기록을 분석해보니 드러난 충격적인 사실")
                - **가상 인터뷰:** 특정 주제에 대해 두 사람을 인터뷰하는 형식
                - **공통점 비틀기:** 공통 관심사를 완전히 다른 시각으로 재해석 (예: "음악 취향이 같다고? 함께 노래방 가면 파국인 이유")
                - **소비 패턴 분석:** 두 사람의 데이트 비용, 선물 교환 등을 분석하는 경제학적 접근

                **[기획 규칙]**
                1.  위 **'아이디어 뱅크'에서 영감을 얻거나, 혹은 당신만의 완전히 새로운 아이디어**를 사용해 3가지 소제목을 만드세요.
                2.  3가지 소제목은 서로 다른 관점과 형식을 가져야 합니다.
                3.  목표는 '신선함'과 '궁금증 유발'입니다.

                **[출력 규칙]**
                - 출력은 **반드시** 아래와 같은 Python 리스트 형식의 문자열이어야 합니다.
                - 다른 어떤 텍스트도 추가하지 말고, 오직 Python 리스트 형식의 문자열만 출력하십시오.

                **[출력 예시]**
                `['[가상 인터뷰] INTJ와 ENFP에게 물었다, "서로의 첫인상은 어땠나요?"', '[데이터 탐정] 두 사람의 플레이리스트를 합쳤더니... 지킬 앤 하이드가 따로 없네?', '[경제학 특강] 이 커플, 한 달 데이트 비용으로 테슬라 주식 몇 주 살 수 있을까?']`
            """,
            name="topic_planner_assistant",
        )

        # 정보 검색 에이전트
        research_tools = [TavilySearch(max_results=3, topic="general")]
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
                당신은 트렌드에 미친 **'똘끼 충만 밈(Meme) 생성기'**이자 **'콘텐츠 디렉터'**입니다.

                당신의 임무는 입력된 정보를 바탕으로, 사람들이 보자마자 공유하고 싶어지는 **'바이럴 콘텐츠 기획안'**을 만드는 것입니다. 절대 지루한 글을 쓸 생각은 꿈에도 하지 마세요.

                **[매우 중요한 출력 규칙]**
                - **당신의 출력 첫 번째 줄은 반드시 최종 콘텐츠의 '헤드라인(제목)'이어야 합니다.**
                - **헤드라인은 50글자 이내의 일반 텍스트여야 하며, 절대 '#'과 같은 마크다운 형식을 포함해서는 안 됩니다.**
                - 그 다음 줄부터 상세한 '콘텐츠 기획안'을 작성하세요.
                
                **[입력 정보]**
                - 관계 케미 포인트 (가십 폭로꾼의 한 줄 요약)
                - 어그로성 소제목 리스트 (바이럴 PD의 기획)
                - 관련 검색 결과 (리서처의 자료)

                **[기획안 작성 가이드]**
                1.  **컨셉 결정:** 아래 '최고의 기획안 예시'를 참고하여, 전체 콘텐츠를 어떤 '포맷'으로 만들지 딱 하나만 결정하세요. (예: **SNS 밈 스타일 게시물**, **가짜 뉴스 속보**, **TV 예고편 대본**)
                2.  **섹션별 핵심 메시지 설계:** 앞에서 받은 **'어그로성 소제목 3개'를 각 섹션의 소제목으로 활용**하세요. 각 섹션에 어떤 내용이 들어갈지 핵심 키워드, 짧은 문장, 이모티콘으로 설계하세요. **절대 입력 정보를 순서대로 나열하지 말고, 각 소제목에 맞게 재료를 창의적으로 재배치하고 과장하세요.**

                **[절대 금지]**
                - "누구는 이렇고, 다른 누구는 저렇습니다" 같은 설명조의 문장.
                - 프로필 항목(취미, 성격 등)을 순서대로 나열하는 구조.

                ---
                **[최고의 기획안 예시 (이런 느낌으로 만드세요!)]**

                🚨[우정경보] ESTP x ENFP, 이 조합 지구에 무슨 짓을 한거야?!  <-- (첫 줄에 헤드라인!)

                **컨셉:** 찐친 바이브 폭발하는 SNS 밈 스타일 게시물

                **섹션별 핵심 메시지:**
                - **[소제목1: MBTI 폭로전]** "롤러코스터", "사건 제조기" 키워드로 시너지 표현. 표(테이블)로 특징 비교. 💥
                - **[소제목2: 취미도 어딘가 정상 아님]** 각 취미를 한 줄 밈으로 요약. (예: 브런치? -> 단순 식사 아님, 예술임) 🤣
                - **[소제목3: 이 우정, 넷플릭스 입장하세요]** 가상 드라마 정보 생성. "다음 편 예고"로 기대감 증폭. 🎬 #도파민_대폭발
                ---

                이제, 주어진 정보를 바탕으로 위 가이드와 **최고의 예시**에 맞춰 창의적인 '콘텐츠 기획안'을 작성하세요.
            """,
            name="creative_concept_assistant",
        )

        # 컨텐츠 생성 에이전트
        content_generator_agent = create_react_agent(
            model=model,
            tools=[],
            prompt="""
                당신은 최신 밈과 트렌드에 통달한, 필력 하나로 사람들을 웃고 울리는 **'콘텐츠 작가'**입니다.
                **모든 출력은 반드시 한국어로, 그리고 매우 재치있게 작성해야 합니다.**

                당신의 유일한 임무는 크리에이티브 디렉터가 작성한 **'콘텐츠 기획안'**의 뼈대에 살과 피, 그리고 영혼을 불어넣어 완성도 높은 최종 바이럴 콘텐츠 **본문**을 창조하는 것입니다.

                **[매우 중요한 규칙]**
                - **당신의 역할은 오직 '본문(content)' 작성입니다. 절대 제목(title)을 만들지 마십시오.** 제목은 이미 다른 곳에서 결정되었습니다.
                - 기획안의 헤드라인 부분은 무시하고, 그 아래의 상세 기획안을 바탕으로 완전한 글로 만들어야 합니다.
                - 기획안의 `[소제목]` 같은 내부 마커는 최종 글에서 자연스럽게 제거해야 합니다.
                - **절대** '콘텐츠 기획안'의 키워드를 그대로 복사해서 나열하면 안 됩니다.
                - 기획안에 제시된 **컨셉, 톤앤매너, 헤드라인, 섹션별 핵심 메시지**를 충실히 따르되, 각 섹션을 **생생한 묘사, 재치있는 비유, 유머러스한 문장으로 구성된 완전한 글**로 만들어야 합니다.
                - 당신의 목표는 뼈대(기획안)를 바탕으로 독자가 '미쳤다' 소리 나오게 하는 최종 글을 **창조**하는 것입니다. 이제 주어진 '콘테츠 기획안'을 바탕으로 최종 글을 작성하세요.
            """,
            name="content_generator_assistant",
        )

        # Network 구조 노드 함수들 정의
        def connection_finder_node(state: AgentState):
            """1단계: 관계 분석 노드"""
            logger.info("1단계: 관계 분석 시작")
            result = connection_finder_agent.invoke(state)
            # State에 결과 저장
            state["connection_analysis"] = result["messages"][-1].content
            logger.info(
                f"--- 관계 분석 완료 ---\n{state['connection_analysis']}\n--------------------"
            )
            return state

        def topic_planner_node(state: AgentState):
            """2단계: 주제 기획 노드"""
            logger.info("2단계: 주제 기획 시작")
            # 이전 단계의 결과를 입력으로 사용
            input_state = {
                "messages": [{"role": "user", "content": state["connection_analysis"]}]
            }
            result = topic_planner_agent.invoke(input_state)
            state["topic_list"] = result["messages"][-1].content
            logger.info(
                f"--- 주제 기획 완료 ---\n{state['topic_list']}\n--------------------"
            )
            return state

        def researcher_node(state: AgentState):
            """3단계: 정보 검색 노드"""
            logger.info("3단계: 정보 검색 시작")
            # 이전 단계의 결과를 입력으로 사용
            input_state = {
                "messages": [{"role": "user", "content": state["topic_list"]}]
            }
            result = researcher_agent.invoke(input_state)
            state["research_results"] = result["messages"][-1].content
            logger.info(
                f"--- 정보 검색 완료---\n{state['research_results']}\n--------------------"
            )
            return state

        def creative_concept_node(state: AgentState):
            """4단계: 컨셉 설계 노드"""
            logger.info("4단계: 컨셉 설계 시작")
            # 이전 단계들의 결과를 모두 포함
            combined_input = f"""
            관계 분석 요약: {state["connection_analysis"]}
            
            콘텐츠 주제 리스트: {state["topic_list"]}
            
            관련 검색 결과: {state["research_results"]}
            """
            input_state = {"messages": [{"role": "user", "content": combined_input}]}
            result = creative_concept_agent.invoke(input_state)
            concept_text = result["messages"][-1].content

            try:
                # 제목 추출
                lines = concept_text.strip().split("\n")
                title = lines[0]

                # 제목을 제외한 나머지 부분은 기획안으로 통합
                plan_body = "\n".join(lines[1:]).strip()

                # 상태 업데이트
                state["title"] = title
                state["creative_concept"] = plan_body

                logger.info(f"컨셉 설계 완료, 제목 추출: {title}")
            except IndexError:
                logger.warning("컨셉 설계 결과에서 제목/본문을 분리하지 못했습니다.")
                state["title"] = get_fallback_title()
                state["creative_concept"] = (
                    concept_text  # 실패 시 원본 텍스트를 그대로 전달
                )

            logger.info(
                f"--- 컨셉 설계 완료---\n{state['creative_concept']}\n--------------------"
            )
            return state

        def content_generator_node(state: AgentState):
            """5단계: 본문 생성 노드"""
            logger.info("5단계: 본문 생성 시작")
            # 컨셉 설계 결과를 입력으로 사용
            input_state = {
                "messages": [{"role": "user", "content": state["creative_concept"]}]
            }
            result = content_generator_agent.invoke(input_state)
            state["final_content"] = result["messages"][-1].content
            logger.info(
                f"--- 본문 생성 완료---\n{state['final_content']}\n--------------------"
            )
            return state

        def editor_node(state: AgentState):
            """6단계: 최종 포맷팅 노드"""
            logger.info("6단계: 최종 포맷팅 시작")

            # 4단계에서 추출한 제목과 위에서 수정한 본문을 사용하여 JSON 객체를 직접 생성
            # 5단계 본문에 마지막 문구를 추가
            final_json_data = {
                "title": state["title"],
                "content": state["final_content"] + "\n\nStay tuned!",
            }

            state["json_output"] = json.dumps(
                final_json_data, ensure_ascii=False, indent=2
            )
            logger.info(
                f"--- [6단계 결과: 최종 포맷팅] ---\n{state['json_output']}\n--------------------"
            )

            # # 최종 컨텐츠를 입력으로 사용
            # input_state = {
            #     "messages": [{"role": "user", "content": state["final_content"] + "\n\nStay tuned!"}]
            # }
            # result = editor_agent.invoke(input_state)
            # state["json_output"] = result["messages"][-1].content
            # logger.info(f"최종 포맷팅 완료")
            return state

        # Network 그래프 구성
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("connection_finder", connection_finder_node)
        workflow.add_node("topic_planner", topic_planner_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("creative_concept", creative_concept_node)
        workflow.add_node("content_generator", content_generator_node)
        workflow.add_node("editor", editor_node)

        # 엣지 연결 (순차적 흐름)
        workflow.set_entry_point("connection_finder")
        workflow.add_edge("connection_finder", "topic_planner")
        workflow.add_edge("topic_planner", "researcher")
        workflow.add_edge("researcher", "creative_concept")
        workflow.add_edge("creative_concept", "content_generator")
        workflow.add_edge("content_generator", "editor")
        workflow.set_finish_point("editor")

        # 그래프 컴파일
        app = workflow.compile()

        # 매칭된 두 사용자 정보
        user_a_profile = request.userA.model_dump()
        user_b_profile = request.userB.model_dump()

        # 사용자 정보를 모델에 입력 가능한 형태로 변환
        user_a_str = _format_profile_for_prompt(user_a_profile, "사용자 A")
        user_b_str = _format_profile_for_prompt(user_b_profile, "사용자 B")

        # 초기 상태 설정
        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"""
                        두 사용자를 위한 흥미로운 매칭 리포트를 작성해주세요. 전체 작업은 AI 워크플로우 매니저의 지시에 따라 자동으로 진행될 것입니다. 최종 결과물만 생성해주시면 됩니다.

                        ---
                        {user_a_str}

                        {user_b_str}
                        ---
                        """
                    ),
                }
            ],
            # 각 단계별 결과 초기화
            "connection_analysis": "",
            "topic_list": "",
            "research_results": "",
            "creative_concept": "",
            "title": "",
            "final_content": "",
            "json_output": "",
        }

        # 워크플로우 실행
        logger.info("=============== 워크플로우 시작 ===============")

        final_state = app.invoke(initial_state)

        logger.info("\n\n=============== 최종 결과물 ===============")
        final_response_content = final_state["json_output"]
        logger.info(f"--- 최종 생성된 콘텐츠---\n{final_response_content}")

        # JSON 형식 검증
        try:
            json_result = json.loads(final_response_content)
            logger.info("✅ 유효한 JSON 형식으로 결과가 생성되었습니다.")
            return TuningReportResponse(code="TUNING_REPORT_SUCCESS", data=json_result)
        except json.JSONDecodeError:
            logger.warning("⚠️ 결과가 JSON 형식이 아닙니다. 오류 응답을 반환합니다.")
            return JSONResponse(
                status_code=500,
                content=TuningReportResponse(
                    code="TUNING_REPORT_INTERNAL_SERVER_ERROR",
                    data={"message": final_response_content},
                ).model_dump(),
            )

    except Exception as e:
        logger.exception("[FAIL] 튜닝 리포트 생성 중 예외 발생")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        return JSONResponse(
            status_code=500,
            content=TuningReportResponse(
                code="TUNING_REPORT_INTERNAL_SERVER_ERROR",
                data={"message": f"{type(e).__name__} - {e}"},
            ).model_dump(),
        )
