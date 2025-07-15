# 튜닝리포트(뉴스) 생성 컨트롤러

from fastapi import HTTPException

from ...schemas.tuning_schema import TuningReport, TuningReportResponse
from ...services.tuning_report_service_gcp_prod import generate_tuning_report_with_agent


async def create_tuning_report(users_data: TuningReport) -> TuningReportResponse:
    try:
        # 멀티에이전트
        result = await generate_tuning_report_with_agent(users_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
