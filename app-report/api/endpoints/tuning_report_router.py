# 튜닝리포트(뉴스) 생성 관련 엔드포인트
from fastapi import APIRouter, Body

from ...schemas.tuning_schema import TuningReport, TuningReportResponse
from ..controllers import tuning_report_controller


class TuningReportRouter:
    """
    사용자 매칭 뉴스 생성 엔드포인트를 처리하는 라우터 클래스
    뉴스 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(prefix="/api", tags=["report"])
        self.router.add_api_route(
            "/v2/report",
            self.create_tuning_report,
            methods=["POST"],
            response_model=TuningReportResponse,
            summary="튜닝 리포트(뉴스) 생성 요청",
            description="해당 사용자 정보를 기반으로 매칭 리포트를 생성합니다.",
        )

    async def create_tuning_report(
        self, users_data: TuningReport = Body(..., description="사용자 관심사 데이터")
    ) -> TuningReportResponse:
        return await tuning_report_controller.create_tuning_report(users_data)
