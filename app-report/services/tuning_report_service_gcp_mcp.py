import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ..core.prompt_templates.tuning_report_prompt import build_prompt
from ..models import qwen_loader_gcp_ollama
from ..schemas.tuning_schema import TuningReport, TuningReportResponse

logger = logging.getLogger(__name__)


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        parent_dir, _ = os.path.split(os.path.dirname(__file__))
        config_path = os.path.join(parent_dir, "config", "mcp_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
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

    return server_config


def extract_json_from_content(content: str) -> dict:
    """
    content에서 JSON 구조 추출 (중첩된 JSON 문자열 처리)
    """
    try:
        # 먼저 직접 JSON 파싱 시도
        parsed = json.loads(content)

        # 파싱된 결과가 title과 content를 가지고 있다면 반환
        if isinstance(parsed, dict) and "title" in parsed and "content" in parsed:
            logger.debug("Success JSON parsing")
            return parsed

    except (json.JSONDecodeError, TypeError):
        logger.debug("Failed JSON parsing")
        pass

    # 정규식으로 JSON 패턴 찾기
    json_pattern = r'\{\s*"title"\s*:\s*"[^"]*"\s*,\s*"content"\s*:\s*"[^"]*"\s*\}'
    match = re.search(json_pattern, content, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.debug("Failed JSON parsing in regular expression")
            pass

    # 더 유연한 패턴으로 시도 (멀티라인 지원)
    json_pattern_flexible = (
        r'\{\s*"title"\s*:\s*"([^"]*?)"\s*,\s*"content"\s*:\s*"(.*?)"\s*\}'
    )
    match = re.search(json_pattern_flexible, content, re.DOTALL)

    if match:
        return {
            "title": match.group(1),
            "content": match.group(2).replace('\\"', '"').replace("\\n", "\n"),
        }

    # 모든 시도 실패 시 기본값 반환
    logger.debug("Failed JSON parsing completely")
    return {
        "title": "매칭 공지",
        "content": content if content else "JSON 파싱 실패",
    }


def build_messages(category: str, userA: dict, userB: dict) -> List[Dict[str, str]]:
    """사용자 데이터를 AI 모델용 메시지로 변환하는 순수 함수"""
    system_prompt = (
        "당신은 데이팅 앱의 작가입니다. 주어진 사용자 정보로 흥미롭고 유쾌한 **공지문 제목과 본문만** 작성해주세요. "
        "기사 외에는 어떤 설명이나 생각도 쓰지 마세요. 기사 내용은 마크다운 형식으로 작성하고 "
        "기사를 json 형식으로 아래처럼 구성하세요:\n\n"
        '{ "title": "기사 제목", "content": "기사 내용" }'
    )

    prompt_text = build_prompt(category=category, userA=userA, userB=userB)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def extract_content_from_response(model_response: Dict[str, Any]) -> str:
    """LangChain 응답에서 텍스트 추출하는 순수 함수"""
    if "messages" in model_response and len(model_response["messages"]) > 0:
        last_message = model_response["messages"][-1]
        if hasattr(last_message, "content"):
            return last_message.content
    else:
        raise ValueError("Invalid messages value")


def parse_json_content(raw_content: str) -> Tuple[str, str]:
    """JSON 문자열을 파싱해서 제목과 내용을 추출하는 순수 함수"""
    try:
        parsed = extract_json_from_content(raw_content)
        title = parsed.get("title", "매칭 공지")
        content = parsed.get("content", "매칭 정보가 생성되었습니다.")
        return title, content
    except Exception as e:
        logger.error(f"JSON 파싱 실패: {e}")
        return "매칭 공지", "매칭 정보 생성 중 오류가 발생했습니다."


class MCPConnectionManager:
    """MCP 클라이언트 연결과 도구 관리를 담당하는 클래스"""

    def __init__(self):
        self._client = None
        self._tools = None
        self._is_initialized = False

    async def get_tools(self) -> List:
        """도구 목록을 반환 (캐싱된 결과 활용)"""
        if not self._is_initialized:
            await self._initialize_connection()
        return self._tools or []

    async def _initialize_connection(self):
        """MCP 연결 초기화 (한 번만 실행)"""
        try:
            server_config = create_server_config()
            self._client = MultiServerMCPClient(server_config)
            self._tools = await self._client.get_tools()
            logger.info(f"MCP 도구 {len(self._tools)}개 로드 성공")
        except Exception as e:
            logger.warning(f"MCP 도구 로드 실패, 기본 모델 사용: {e}")
            self._tools = []
        finally:
            self._is_initialized = True


class AIAgentManager:
    """AI 에이전트 인스턴스 관리 클래스"""

    def __init__(self, connection_manager: MCPConnectionManager):
        self.connection_manager = connection_manager
        self._agent = None

    async def get_agent(self):
        """에이전트 인스턴스 반환 (싱글톤 패턴)"""
        if self._agent is None:
            tools = await self.connection_manager.get_tools()
            self._agent = create_react_agent(qwen_loader_gcp_ollama.get_model(), tools)
        return self._agent


class TuningReportService:
    """튜닝 리포트 생성의 전체 흐름을 조율하는 서비스"""

    def __init__(self):
        self.connection_manager = MCPConnectionManager()
        self.agent_manager = AIAgentManager(self.connection_manager)

    async def generate_report(self, request: TuningReport) -> TuningReportResponse:
        """메인 비즈니스 로직 - 상태 관리와 순수 함수들을 조합"""
        try:
            # 1. 상태 관리: 에이전트 획득 (재사용)
            agent = await self.agent_manager.get_agent()

            # 2. 순수 함수: 메시지 구성
            messages = build_messages(request.category, request.userA, request.userB)

            # 3. 상태 관리: AI 모델 호출
            model_response = await agent.ainvoke({"messages": messages})
            if not model_response or not isinstance(model_response, dict):
                logger.warning(
                    f"유효하지 않은 model_response 타입: {type(model_response)}"
                )
                raise ValueError(f"'{model_response}' must be a dictionary")
            logger.info("AI 모델 응답 수신 완료")

            # 4. 순수 함수: 응답 처리 파이프라인
            raw_content = extract_content_from_response(model_response)
            title, content = parse_json_content(raw_content)

            return TuningReportResponse(
                code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
            )

        except Exception as e:
            logger.error(f"튜닝 리포트 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "TUNING_REPORT_INTERNAL_SERVER_ERROR",
                    "message": str(e),
                },
            )
