"""
성능 모니터링 및 지표 요약 API 엔드포인트 정의 및 관리
수집된 성능 로그를 기반으로 성능 요약 통계를 제공
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from utils import logger


class PerformanceRouter:
    """
    성능 모니터링 및 요약 통계 관련 엔드포인트를 처리하는 라우터 클래스
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(
            prefix="/monitoring", tags=["monitoring(내부 로그 확인용)"]
        )
        # 엔드포인트 등록 (/monitoring/performance-summary)
        self.router.add_api_route(
            "/performance-summary",
            self.get_summary,
            methods=["GET"],
            summary="성능 지표 요약 조회",
            description="API 응답 시간, 메모리 사용량, 오류 횟수 등 애플리케이션 성능 관련 메트릭 요약 정보를 조회합니다.",
        )

    def get_summary(self) -> JSONResponse:
        """
        수집된 성능 지표 요약 정보를 반환

        **응답 예시**:
        ```json
        {
          "code": "PERFORMANCE_SUMMARY_RETRIEVED",
          "data": {
            "api_response_times": {...},
            "embedding_generation": {...},
            ...
          }
        }
        ```
        """
        summary = logger.get_performance_summary()
        return JSONResponse(
            content={"code": "PERFORMANCE_SUMMARY_RETRIEVED", "data": summary}
        )
