import asyncio
import json
import operator
import os
import re
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent

# from langgraph.prebuilt import create_react_agent
# from langgraph_supervisor import create_supervisor
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from ...models import qwen_loader_gcp_vllm
from ...utils.logger import logger


def safe_json_parse(text: str) -> dict:
    """JSON 문자열에서 코드 블록과 설명을 제거하고 안전하게 파싱합니다."""
    # ```json ... ``` 패턴을 찾아 내용만 추출
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)

    # 간혹 응답이 {..} 형태가 아닌 경우가 있어 추가 보정
    text = text.strip()
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}\n원본 텍스트: {text}")
        return {}  # 실패 시 빈 딕셔너리 반환


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        # parent_dir, _ = os.path.split(os.path.dirname(__file__))
        # config_path = os.path.join(parent_dir, "config", "mcp_config.json")
        current_file_path = os.path.abspath(__file__)
        app_report_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(current_file_path))
        )
        config_path = os.path.join(app_report_dir, "config", "mcp_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return {}


def create_server_config():
    """MCP 서버 설정을 생성합니다."""
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # command가 있으면 stdio 방식
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio",
                }
            # url이 있으면 sse 방식
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }
    if not server_config:
        logger.warning("⚠ MCP 서버 설정이 비어 있습니다. MCP 기능이 비활성화됩니다.")

    return server_config


# --- StateGraph 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # --- [핵심] 각 단계의 결과물을 저장할 필드 추가 ---
    connection_summary: str  # 1단계: 관계 분석 요약문
    topic_list: list  # 2단계: 주제 리스트
    research_summary: str  # 3단계: 리서치 결과
    creative_concept: str  # 4단계: 컨셉 기획안
    final_script: dict  # 5단계: 최종 글 (title, content)


class ReportGenerationPipeline:
    def __init__(self):
        # 클래스 초기화 시 모델을 로드합니다.
        self.model = qwen_loader_gcp_vllm.get_model()
        self.graph = self._build_graph()

    # 관계 분석 에이전트
    async def connection_finder_node(self, state: AgentState) -> dict:
        logger.info("--- [Node] Connection Finder 실행 ---")
        user_input = state["messages"][0].content
        system_prompt = """
            당신은 고도로 훈련된 '관계 분석 전문가'입니다.

            당신의 유일한 임무는 입력으로 주어진 두 사용자의 프로필 데이터를 바탕으로, 두 사람의 **공통점**과 **상호보완적인 특징**을 찾아내어 한 문단의 간결한 텍스트로 요약하는 것입니다.

            **규칙:**
            - 절대 입력된 프로필 정보 외의 내용을 추측하거나 만들어내지 마십시오.
            - 분석 결과는 다음 전문가가 쉽게 이해할 수 있도록 서술형 문장으로 작성해야 합니다.
            - 다른 인사나 부가 설명 없이, 분석 요약문만 즉시 출력하십시오.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = await self.model.ainvoke(messages)
        return {"connection_summary": response.content}

    async def topic_planner_node(self, state: AgentState) -> dict:
        logger.info("--- [Node] Topic Planner 실행 ---")
        connection_summary = state["connection_summary"]
        system_prompt = """
                당신은 창의적인 '콘텐츠 기획자'입니다.

                당신의 임무는 이전 단계에서 전달받은 '관계 분석 요약문'을 읽고, 그 내용을 기반으로 흥미로운 콘텐츠 주제 3개를 제안하는 것입니다.

                **규칙:**
                - 제안하는 주제는 반드시 분석된 내용(공통점, 특징)과 관련 있어야 합니다.
                - 출력은 **반드시** 아래와 같은 Python 리스트 형식의 문자열이어야 합니다.
                - 예시 형식: `['두 사람의 MBTI 궁합 파헤치기', '공통 관심사인 영화 취향으로 본 데이트 코스', '서로 다른 입맛을 모두 만족시킬 맛집 탐방 코스']`
                - 다른 어떤 텍스트도 추가하지 말고, 오직 Python 리스트 형식의 문자열만 출력하십시오.
            """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": connection_summary},
        ]

        response = await self.model.ainvoke(messages)
        # 작은 모델은 형식을 잘 못 지킬 수 있으므로, 응답에서 리스트 부분만 추출하는 로직 추가
        content = response.content
        try:
            # 응답 안에 있는 리스트 문자열을 찾습니다.
            list_str = content[content.find("[") : content.rfind("]") + 1]
            topic_list = json.loads(list_str.replace("'", '"'))
        except (ValueError, json.JSONDecodeError):
            logger.error(
                f"Topic Planner가 리스트 형식을 생성하지 못했습니다. 응답: {content}"
            )
            topic_list = ["모델이 주제 생성에 실패했습니다."]  # 실패 시 기본값

        return {"topic_list": topic_list}

    # --- [핵심] researcher_agent를 대체할 새로운 노드 함수 ---
    async def researcher_node(self, state: AgentState) -> dict:
        logger.info("--- [Node] Researcher 실행 ---")
        topics = state["topic_list"]

        server_config = create_server_config()
        client = MultiServerMCPClient(server_config)
        try:
            tools = await client.get_tools()
            logger.debug("MCP 툴 개수: ", len(tools))
        except Exception as e:
            logger.warning(f"[INFO] MCP 도구 로드 실패 또는 초기화 안됨: {e}")
            tools = []

        prompt = hub.pull("hwchase17/structured-chat-agent")
        agent_runnable = create_structured_chat_agent(self.model, tools, prompt)
        executor = AgentExecutor(
            agent=agent_runnable, tools=tools, handle_parsing_errors=True, verbose=True
        )

        # system_prompt="""
        #     당신은 오직 'tavily_web_search' 도구만 사용하는 '최신 트렌드 리서처'입니다.
        #     **모든 출력은 반드시 한국어로 작성해야 합니다.**

        #     당신의 임무는 이전 단계에서 받은 각 주제에 대해, **반드시 `tavily_web_search` 도구를 사용하여** 웹에서 최신 정보를 검색하고 요약하는 것입니다.

        #     **절대 규칙:**
        #     1.  **절대 당신의 내부 지식이나 의견으로 답변하지 마십시오.** 당신은 창작가가 아니라 검색기입니다.
        #     2.  입력받은 주제 리스트의 각 항목에 대해 **하나씩, 개별적으로** `tavily_web_search` 도구를 호출해야 합니다.
        #     3.  도구 호출 후 받은 검색 결과를, 다음 작가가 사용할 수 있도록 **출처(URL)와 함께 핵심 내용만** 간결한 불렛 포인트(`-`)로 요약하여 정리하십시오.
        # """

        all_results = []
        for topic in topics:
            logger.info(f"'{topic}'에 대한 검색을 시작합니다...")

            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": topic},
            # ]

            input_prompt = f"다음 주제에 대해 웹에서 정보를 검색하고, 그 결과를 한국어로 상세히 요약해줘: '{topic}'"

            try:
                response = await executor.ainvoke({"input": input_prompt})
                topic_result = response.get(
                    "output", f"'{topic}'에 대한 검색 결과가 없습니다."
                )
                all_results.append(f"- 주제 '{topic}' 검색 결과:\n{topic_result}")
            except Exception as e:
                error_msg = f"'{topic}' 검색 중 오류 발생: {e}"
                logger.error(error_msg)
                all_results.append(f"- 주제 '{topic}' 검색 실패:\n{error_msg}")

        # 7. 결과를 LangGraph 상태에 맞게 포맷팅합니다.
        research_summary = "\n\n".join(all_results)
        return {"research_summary": research_summary}

    async def creative_concept_node(self, state: AgentState) -> dict:
        logger.info("--- [Node] Creative Concept 실행 ---")
        # 모든 이전 결과물을 조합하여 프롬프트를 만듭니다.
        prompt_input = f"""
        [관계 분석]: {state['connection_summary']}
        [기획 주제]: {state['topic_list']}
        [리서치 결과]: {state['research_summary']}
        
        위 정보를 바탕으로 '콘텐츠 기획안'을 작성해주세요.
        """
        system_prompt = """
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
            """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_input},
        ]
        # creative_concept_prompt를 사용하여 LLM 호출
        response = await self.model.ainvoke(messages)
        return {"creative_concept": response.content}

    # 컨텐츠 생성 에이전트
    async def content_generator_node(self, state: AgentState) -> dict:
        logger.info("--- [Node] Content Generator 실행 ---")
        # creative_concept 결과를 입력으로 사용
        creative_concept = state["creative_concept"]
        system_prompt = """
                당신은 뛰어난 필력을 가진 '콘텐츠 작가'입니다.
                **모든 출력은 반드시 한국어로 작성해야 합니다.**

                당신의 유일한 임무는, 이전 단계의 크리에이티브 디렉터가 작성한 **'콘텐츠 기획안'을 입력으로 받아, 그 뼈대에 살을 붙여 재미있고 완성도 높은 최종 글을 작성**하는 것입니다.

                **매우 중요한 규칙:**
                - **절대** '콘텐츠 기획안'을 그대로 복사해서는 안 됩니다.
                - 기획안에 제시된 **컨셉, 톤앤매너, 헤드라인, 섹션별 핵심 메시지**를 충실히 따르면서, 각 섹션을 **자연스러운 문장과 풍부한 묘사가 담긴 완전한 문단**으로 만들어야 합니다.
                - 당신의 목표는 뼈대(기획안)를 바탕으로 멋진 최종 글을 **창조**하는 것입니다. 이제 주어진 '콘텐츠 기획안'을 바탕으로 최종 글을 작성하세요.
            """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": creative_concept},
        ]

        # ... LLM 호출 ...
        response = await self.model.ainvoke(messages)
        final_script = safe_json_parse(response.content)
        logger.info(f"Generated Script Title: {final_script.get('title')}")
        return {"final_script": final_script}

    # 검수 및 편집 에이전트
    async def editor_node(self, state: AgentState) -> dict:
        """
        content_generator가 생성한 글을 최종적으로 검수하고,
        완벽한 JSON 형식으로 포맷팅하는 최종 관문 역할.
        """
        logger.info("--- [Node 6] Editor (Final QA) 실행 ---")

        # 1. 이전 단계에서 생성된 'final_script' 딕셔너리를 가져옵니다.
        #    이때 get()을 사용하여, 만약 키가 없어도 에러가 나지 않도록 합니다.
        script_dict = state.get("final_script", {})

        # 2. content_generator가 JSON 파싱에 실패하여 빈 dict를 전달했을 경우에 대비.
        if (
            not script_dict
            or not script_dict.get("title")
            or not script_dict.get("content")
        ):
            error_content = (
                "콘텐츠 생성 단계에서 유효한 제목과 본문을 만들지 못했습니다."
            )
            logger.error(error_content)
            # 최종 출력도 JSON 형식을 유지해주는 것이 좋습니다.
            final_json_str = json.dumps(
                {"title": "생성 오류", "content": error_content},
                ensure_ascii=False,
                indent=2,
            )
            return {
                "messages": [AIMessage(content=final_json_str, name="editor_assistant")]
            }

        # 3. LLM에게 최종 포맷팅을 지시하는 프롬프트를 구성합니다.
        #    이전 단계의 결과(딕셔너리)를 문자열로 변환하여 전달합니다.
        input_text = (
            f"제목: {script_dict.get('title')}\n\n본문:\n{script_dict.get('content')}"
        )

        system_prompt = """
        [ 역할 ] 당신은 출판사의 최종 편집자(Editor)이자, 매우 엄격한 '데이터 포맷터'입니다.
        [ 지시 ] 모든 답변은 반드시, 그리고 오직 한국어로만 작성해야 합니다.

        당신의 유일한 임무는 입력으로 받은 '제목'과 '본문' 텍스트를 **단 하나의, 다른 어떤 설명도 없는, 완벽한 JSON 객체**로 만드는 것입니다.

        **절대 규칙:**
        1.  입력 텍스트의 내용을 창의적으로 수정하지 마십시오. 오탈자나 문법 교정은 가능합니다.
        2.  당신의 최종 출력물은 **오직 JSON 객체 하나**여야 합니다. "다음은 JSON입니다" 와 같은 설명, 인사, 코드 블록 마크다운(```) 등 **JSON 외의 문자는 단 하나도 포함해서는 안 됩니다.**
        3.  최종 출력 형식: `{"title": "입력받은 제목", "content": "입력받은 본문 내용"}`
        4.  본문 내용(`content`)의 모든 줄바꿈은 `\\n`으로 이스케이프 처리해야 합니다.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_text),
        ]

        # 4. LLM 호출
        response = await self.model.ainvoke(messages)

        # 5. LLM이 생성한 최종 결과물을 messages에 담아 반환합니다.
        #    여기서는 LLM이 완벽한 JSON 문자열을 생성했다고 가정합니다.
        #    만약 여기서도 파싱 에러가 발생한다면, 추가적인 예외 처리가 필요할 수 있습니다.
        final_json_str = response.content
        logger.info("Editor가 최종 JSON 포맷팅 완료.")

        return {
            "messages": [AIMessage(content=final_json_str, name="editor_assistant")]
        }

    def _build_graph(self) -> StateGraph:
        # --- StateGraph로 결정론적 파이프라인 구축 ---
        workflow = StateGraph(AgentState)

        # 각 노드를 그래프에 추가
        workflow.add_node("connection_finder", self.connection_finder_node)
        workflow.add_node("topic_planner", self.topic_planner_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("creative_concept", self.creative_concept_node)
        workflow.add_node("content_generator", self.content_generator_node)
        workflow.add_node("editor", self.editor_node)

        # 엣지를 순서대로 연결 (이제 분기 로직이 필요 없음)
        workflow.add_edge(START, "connection_finder")
        workflow.add_edge("connection_finder", "topic_planner")
        workflow.add_edge("topic_planner", "researcher")
        workflow.add_edge("researcher", "creative_concept")
        workflow.add_edge("creative_concept", "content_generator")
        workflow.add_edge("content_generator", "editor")
        workflow.add_edge("editor", END)

        # 그래프 컴파일 및 실행
        return workflow.compile()

    async def run(self):
        """파이프라인을 실행하고 결과를 출력합니다."""
        initial_input = {
            "messages": [
                HumanMessage(
                    content="""
                    두 사용자를 위한 흥미로운 매칭 리포트를 작성해주세요. 전체 작업은 자동으로 진행될 것입니다.

                    ---
                    **사용자 A 프로필:**
                    - 이름: 김지훈, MBTI: INTP, 관심사: 음악 감상, 넷플릭스 보기, 코딩, 보드게임

                    **사용자 B 프로필:**
                    - 이름: 박서연, MBTI: ENFJ, 관심사: 음악 감상, 맛집 탐방, 봉사활동
                    ---
                    """
                )
            ]
        }
        final_node_output = None
        async for chunk in self.graph.astream(initial_input):
            logger.debug(f"--- 스트리밍 청크 ---\n{chunk}")
            if END in chunk:
                final_node_output = chunk[END]

        logger.info("\n\n=============== 최종 결과물 ===============")
        if final_node_output:
            final_message_content = final_node_output["messages"][-1].content
            logger.info(f"최종 생성된 콘텐츠:\n{final_message_content}")
        else:
            logger.error(
                "최종 결과물을 찾을 수 없습니다. (그래프가 정상 종료되지 않음)"
            )


if __name__ == "__main__":
    pipeline = ReportGenerationPipeline()
    asyncio.run(pipeline.run())
