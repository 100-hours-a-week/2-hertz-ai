# 튜닝리포트(뉴스) 생성 컨트롤러
from fastapi import HTTPException
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse
from services.chat_report_service import handle_chat_reports
from utils.logger import logger


async def handle_chat_report(body: ChatReportRequest) -> ChatReportResponse:

    result = await handle_chat_reports(body)

    if isinstance(result, ChatReportResponse):
        return result

    elif isinstance(result, dict):
        try:
            return ChatReportResponse(**result)
        except Exception as e:
            logger.exception(
                "[handle_chat_report] Failed to unpack dict into ChatReportResponse"
            )
            raise HTTPException(status_code=500, detail="Invalid response format.")

    else:
        logger.warning(f"[handle_chat_report] Unexpected result type: {type(result)}")
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response type from handle_chat_reports: {type(result)}",
        )
