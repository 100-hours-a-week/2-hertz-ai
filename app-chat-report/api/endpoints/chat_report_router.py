# 채팅 신고 판단 기능 엔드포인트
from api.controllers.chat_report_controller import handle_chat_report
from fastapi import APIRouter
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse

router = APIRouter(prefix="/api/v3/chat", tags=["Chat Report"])


"""
채팅 메시지에 대한 신고 요청을 처리하는 라우터 클래스입니다.

입력: 채팅 내용(`message`) #, 신고 사유(`reason`)  
출력: 해당 발언이 신고 사유에 부합할 경우 `true` (부적절함), 아니라면 `false` (적절함)
"""
router = APIRouter(prefix="/api/v3/chat", tags=["Chat Report"])


@router.post(
    "/report",
    response_model=ChatReportResponse,
    summary="채팅 메시지 신고 제재 판단 요청",
    description="신고 메시지와 사유를 바탕으로 제재 여부를 자동 판단합니다.",
)
async def judge_chat_report(body: ChatReportRequest):
    return await handle_chat_report(body)
