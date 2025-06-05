# my_server.py
import asyncio  # 클라이언트에 나중에 사용할 것입니다.

from fastmcp import FastMCP

# 서버의 이름을 지정하여 인스턴스화합니다.
# mcp = FastMCP(name="내 첫 MCP 서버")

mcp = FastMCP(
    name="구성된 서버",
    port=8080,  # 기본 SSE 포트 설정
    host="127.0.0.1",  # 기본 SSE 호스트 설정
    log_level="DEBUG",  # 로깅 수준 설정
    on_duplicate_tools="warn",  # 같은 이름의 도구가 등록되면 경고 (옵션: 'error', 'warn', 'ignore')
)

print("FastMCP 서버 객체가 생성되었습니다.")


@mcp.tool()
def greet(name: str) -> str:
    """간단한 인사 말합니다."""
    return f"안녕하세요, {name}님!"


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더합니다."""
    return a + b


print("도구 'greet'와 'add'가 추가되었습니다.")

APP_CONFIG = {"theme": "dark", "version": "1.1", "feature_flags": ["new_dashboard"]}


@mcp.resource("data://config")
def get_config() -> dict:
    """애플리케이션 구성을 제공합니다."""
    return APP_CONFIG


print("리소스 'data://config'가 추가되었습니다.")

USER_PROFILES = {
    101: {"name": "앨리스", "status": "active"},
    102: {"name": "밥", "status": "inactive"},
}


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """사용자의 ID로 사용자 프로필을 검색합니다."""
    # URI에서의 {user_id}가 자동으로 인수로 전달됩니다.
    return USER_PROFILES.get(user_id, {"error": "사용자를 찾을 수 없습니다."})


print("리소스 템플릿 'users://{user_id}/profile'가 추가되었습니다.")


@mcp.prompt("summarize")
async def summarize_prompt(text: str) -> list[dict]:
    """제공된 텍스트를 요약하는 프롬프트를 생성합니다."""
    return [
        {"role": "system", "content": "당신은 요약에 능숙한 유용한 조수입니다."},
        {"role": "user", "content": f"다음 텍스트를 요약해 주세요:\n\n{text}"},
    ]


print("프롬프트 'summarize'가 추가되었습니다.")

if __name__ == "__main__":
    print("\n--- __main__을 통해 FastMCP 서버 시작 중 ---")
    # 이는 서버를 시작합니다. 일반적으로 기본적으로 stdio 전송을 사용합니다.
    mcp.run()
