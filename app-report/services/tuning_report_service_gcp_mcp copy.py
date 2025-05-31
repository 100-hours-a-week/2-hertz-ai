import json
import logging
import os
import re

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
            return parsed

    except (json.JSONDecodeError, TypeError):
        pass

    # 정규식으로 JSON 패턴 찾기
    json_pattern = r'\{\s*"title"\s*:\s*"[^"]*"\s*,\s*"content"\s*:\s*"[^"]*"\s*\}'
    match = re.search(json_pattern, content, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
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
    return {
        "title": "매칭 공지",
        "content": content if content else "매칭 정보가 생성되었습니다.",
    }


async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    server_config = create_server_config()
    client = MultiServerMCPClient(server_config)
    try:
        tools = await client.get_tools()
    except Exception as e:
        logger.warning(f"MCP 도구 로드 실패, 기본 모델 사용: {e}")
        tools = []
    agent = create_react_agent(qwen_loader_gcp_ollama.get_model(), tools)

    try:
        # 프롬프트 생성
        prompt_text = build_prompt(
            category=request.category, userA=request.userA, userB=request.userB
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 데이팅 앱의 작가입니다. 주어진 사용자 정보로 흥미롭고 유쾌한 **공지문 제목과 본문만** 작성해주세요. "
                    "기사 외에는 어떤 설명이나 생각도 쓰지 마세요. 기사 내용은 마크다운 형식으로 작성하고 "
                    "기사를 json 형식으로 아래처럼 구성하세요:\n\n"
                    '{ "title": "기사 제목", "content": "기사 내용" }'
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        model_response = await agent.ainvoke({"messages": messages})
        print("response: ", model_response)

        # LangChain 응답에서 content 추출
        raw_content = ""
        if "messages" in model_response and len(model_response["messages"]) > 0:
            last_message = model_response["messages"][-1]
            if hasattr(last_message, "content"):
                raw_content = last_message.content

        print("Extracted raw_content:", raw_content)

        # 개선된 JSON 추출 함수 사용
        parsed = extract_json_from_content(raw_content)
        title = parsed.get("title", "매칭 공지")
        content = parsed.get("content", "매칭 정보가 생성되었습니다.")

        # 디버깅 출력
        print(f"Final title: {title}")
        print(f"Final content: {content}")

        response = TuningReportResponse(
            code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
        )

        return response
    except Exception as e:
        logger.error(f"Error generating tuning report: {str(e)}")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        raise e
