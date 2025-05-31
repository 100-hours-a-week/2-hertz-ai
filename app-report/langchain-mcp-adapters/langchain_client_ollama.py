import json

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatOllama(model="qwen2.5:3b")


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        with open("app-report/langchain-mcp-adapters/mcp_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None


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


async def main():
    server_config = create_server_config()
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    # # 단일 답변
    # query = "오늘 서울 날씨 어때?"
    # response = await agent.ainvoke({"messages": query})
    # print(f"도구 호출 결과")
    # print(f"{response['messages'][-1].content}")

    while True:
        user_input = input("질문을 입력하세요: ")
        try:
            response = await agent.ainvoke({"messages": user_input})
            print(f"도구 호출 결과")
            print(f"{response['messages'][-1].content}")
        except Exception as e:
            print(f"에러 발생: {e}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
