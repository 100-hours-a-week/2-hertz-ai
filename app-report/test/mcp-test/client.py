# my_client.py
import asyncio

from fastmcp import Client


async def interact_with_server():
    print("--- 클라이언트 생성 중 ---")

    # 옵션 1: `python my_server.py`를 통해 실행 중인 서버에 연결 (stdio 사용)
    client = Client("app-report/mcp-test/server.py")

    # 옵션 2: `fastmcp run ... --transport sse --port 8080`를 통해 실행 중인 서버에 연결
    # client = Client("http://localhost:8080")  # 정확한 URL/포트를 사용하십시오.

    print(f"클라이언트가 연결될 대상: {client}")

    try:
        async with client:
            print("--- 클라이언트 연결됨 ---")
            # 'greet' 도구 호출
            greet_result = await client.call_tool("greet", {"name": "원격 클라이언트"})
            print(f"greet 결과: {greet_result}")

            # 'config' 리소스 읽기
            config_data = await client.read_resource("data://config")
            print(f"config 리소스: {config_data}")

            # 사용자 프로필 102 읽기
            profile_102 = await client.read_resource("users://102/profile")
            print(f"사용자 102 프로필: {profile_102}")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("--- 클라이언트 상호작용 완료 ---")


if __name__ == "__main__":
    asyncio.run(interact_with_server())
