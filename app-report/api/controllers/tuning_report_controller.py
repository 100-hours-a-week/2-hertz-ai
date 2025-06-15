# 튜닝리포트(뉴스) 생성 컨트롤러

from fastapi import HTTPException

from ...schemas.tuning_schema import TuningReport, TuningReportResponse
from ...services.tuning_report_service_gcp_mcp_prod import generate_tuning_report

# from ...services.tuning_report_service_gcp_mcp import TuningReportService


async def create_tuning_report(users_data: TuningReport) -> TuningReportResponse:
    try:
        # report_service = TuningReportService()
        # result = await report_service.generate_report(users_data)
        result = await generate_tuning_report(users_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
