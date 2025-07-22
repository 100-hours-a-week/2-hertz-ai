import asyncio
from typing import Any

from langchain_core.tools import BaseTool


class AsyncToolWrapper(BaseTool):
    """
    비동기 전용 도구를 동기적으로 호출할 수 있도록 감싸는 래퍼 클래스.
    """

    tool: BaseTool

    # pydantic v2 호환성을 위한 설정
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tool: BaseTool):
        # 래퍼의 이름, 설명, args_schema를 원본 도구와 동일하게 설정
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            tool=tool,
        )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        동기 호출이 들어왔을 때, 비동기 _arun을 이벤트 루프에서 실행.
        """
        # 현재 실행 중인 이벤트 루프를 가져옵니다.
        # 루프가 없으면 새로 생성합니다.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 비동기 함수를 동기적으로 실행하고 결과를 기다립니다.
        return loop.run_until_complete(self.tool._arun(*args, **kwargs))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """비동기 호출은 원본 도구의 _arun을 그대로 실행."""
        return await self.tool._arun(*args, **kwargs)
