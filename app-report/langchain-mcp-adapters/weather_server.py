import os
from typing import List

import aiohttp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for a given location using wttr.in."""
    normalized_location = location.lower()

    url = f"https://wttr.in/{normalized_location}?format=3&lang=ko"

    try:
        print(f"Fetching weather for {location}...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_text = await response.text()
                    return weather_text
                else:
                    return f"{location} 날씨 정보를 불러오지 못했습니다. (HTTP {response.status})"
    except Exception as e:
        return f"날씨 요청 중 오류가 발생했습니다: {e}"


if __name__ == "__main__":
    mcp.run(transport="sse")
