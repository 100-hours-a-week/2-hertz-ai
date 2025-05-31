import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def main():
    model = ChatOllama(model="qwen2.5:3b")
    # model = ChatOllama(model="llama3.2:3b")

    client = MultiServerMCPClient(
        {
            "introduce": {
                "command": "python",
                "args": [
                    "/Users/yoh/workspace/2-hertz-ai/app-report/langchain-mcp-adapters/introduce.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    while True:
        user_input = input("질문을 입력하세요: ")
        try:
            response = await agent.ainvoke({"messages": user_input})
            final_answer = [
                msg
                for msg in response["messages"]
                if msg.__class__.__name__ == "AIMessage" and msg.content
            ][-1].content
            print(f"답변: {final_answer}\n")
        except Exception as e:
            print(f"에러 발생: {e}\n")


asyncio.run(main())
