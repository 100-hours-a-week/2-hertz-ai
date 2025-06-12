import asyncio  # New! Add asyncio to define `main()` as an async function
import os

from dotenv import load_dotenv

# from langchain import hub
# from langchain.agents import AgentExecutor
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": [
                "/Users/yoh/workspace/2-hertz-ai/app-report/mcp-test/math_server.py"
            ],  # make sure you set the correct path
            "transport": "stdio",
        },
        "weather": {
            # make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",  # make sure you opened the localhost
            "transport": "streamable_http",
        },
    }
)

# model = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.5,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
# )


# model = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Qwen 모델 대신 다른 모델 사용
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.5,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
# )

# chat_model = ChatHuggingFace(llm=model)


# Define `tools` and each response that use `await` under the async `main()`
async def main():
    tools = await client.get_tools()
    # prompt = hub.pull("hwchase17/react")
    agent = create_react_agent("google_genai:gemini-2.0-flash", tools)
    # agent = create_react_agent(chat_model, tools)
    # agent_executor = AgentExecutor(
    #     agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    # )

    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(math_response)
    weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    print(weather_response)

    math_final_answer = math_response["messages"][-1].content
    print("Math Answer:", math_final_answer)
    weather_final_answer = weather_response["messages"][-1].content
    print("Weather Answer:", weather_final_answer)


if __name__ == "__main__":
    asyncio.run(main())
