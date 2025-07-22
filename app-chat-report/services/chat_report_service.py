from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from db.mongodb import mongodb, save_report_to_db
from fastapi import HTTPException
from models.hyper_clover_loader import ModelSingleton
from models.kcelectra_base_loader import get_model
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse
from utils.logger import log_performance, logger

message_filter_collection = mongodb.get_collection("message_filters")
# =================================================================
# 설정 및 상수
# =================================================================


class ToxicityLabel(Enum):
    SAFE = "LABEL_0"
    RULE_BASED_PROFANITY = "RULE_BASED_PROFANITY"


@dataclass
class ToxicityConfig:
    CONFIDENCE_THRESHOLD: float = 0.75
    TIMEZONE_OFFSET_HOURS: int = 9

    # 룰베이스 욕설 블랙리스트
    # 이 목록에 포함된 단어는 AI 모델을 거치지 않고 즉시 '유해'로 판단됩니다.
    PROFANITY_BLACKLIST: frozenset = frozenset()


config = ToxicityConfig()

# =================================================================
# 데이터 클래스
# =================================================================


@dataclass
class ToxicityResult:
    is_toxic: bool
    label: str
    confidence: float
    monitoring_required: bool


# =================================================================
# 유틸리티 함수들
# =================================================================


def get_seoul_time() -> str:
    """서울 시간으로 현재 시각을 ISO 형식으로 반환"""
    return (
        (datetime.utcnow() + timedelta(hours=config.TIMEZONE_OFFSET_HOURS))
        .replace(microsecond=0)
        .isoformat()
    )


def should_monitor(confidence: float) -> bool:
    """신뢰도 점수에 따라 모니터링 필요성 판단"""
    return confidence < config.CONFIDENCE_THRESHOLD


async def get_active_profanity_words() -> frozenset[str]:
    """MongoDB에서 활성화된 욕설 목록을 로드"""
    try:
        # Fetch only active words
        cursor = message_filter_collection.find(
            {"is_active": True}, {"word": 1, "_id": 0}
        )
        words = {doc["word"] for doc in await cursor.to_list(length=None)}
        logger.info(f"Loaded {len(words)} active profanity words from DB.")
        return frozenset(words)
    except Exception as e:
        logger.error(f"Failed to load profanity words from DB: {str(e)}")
        # Fallback or raise error, depending on desired behavior
        return frozenset()  # Return empty set to prevent errors, or raise HTTPException


# =================================================================
# 유해성 탐지 함수들
# =================================================================
async def check_rule_based_toxicity(message: str) -> Optional[ToxicityResult]:
    """룰베이스 욕설 검사 (MongoDB 연동)"""
    active_profanity_words = await get_active_profanity_words()
    for profanity in active_profanity_words:
        if profanity in message:
            logger.info(f"Rule-based detection triggered for word: {profanity}")
            return ToxicityResult(
                is_toxic=True,
                label=ToxicityLabel.RULE_BASED_PROFANITY.value,
                confidence=1.0,
                monitoring_required=False,
            )
    return None


def check_ai_model_toxicity(message: str) -> ToxicityResult:
    """AI 모델을 통한 유해성 검사"""
    try:
        model = get_model()
        result = model(message)[0]

        label = result.get("label", ToxicityLabel.SAFE.value)
        confidence_raw = result.get("score", 0.0)
        confidence = round(float(confidence_raw), 4) if confidence_raw else 0.0

        is_toxic = label != ToxicityLabel.SAFE.value
        monitoring_required = should_monitor(confidence)

        return ToxicityResult(
            is_toxic=is_toxic,
            label=label,
            confidence=confidence,
            monitoring_required=monitoring_required,
        )
    except Exception as e:
        logger.error(f"AI model toxicity check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")


async def analyze_toxicity(message: str) -> ToxicityResult:
    """통합 유해성 분석 (룰베이스 + AI 모델)"""
    # 1차: 룰베이스 검사
    rule_result = await check_rule_based_toxicity(message)
    if rule_result:
        return rule_result

    # 2차: AI 모델 검사
    return check_ai_model_toxicity(message)


def create_report_document(
    message_id: str, message_content: str, result: ToxicityResult
) -> Dict[str, Any]:
    """MongoDB에 저장할 문서 생성"""
    return {
        "messageId": message_id,
        "messageContent": message_content,
        "result": result.is_toxic,
        "label": result.label,
        "confidence": result.confidence,
        "monitoring": "Y" if result.monitoring_required else "N",
        "report_time": get_seoul_time(),
    }


def create_response(result: ToxicityResult) -> ChatReportResponse:
    """API 응답 객체 생성"""
    return ChatReportResponse(
        code="CENSORED_SUCCESS",
        data={
            "result": result.is_toxic,
            "label": result.label,
            "confidence": result.confidence,
            "monitoring": "Y" if result.monitoring_required else "N",
        },
    )


# =================================================================
# 메인 처리 함수들
# =================================================================


@log_performance(operation_name="handle_chat_report(clova)", include_memory=True)
async def handle_chat_report_(message: str) -> ChatReportResponse:
    """언어모델 메시지 유해성 검사 (레거시 함수)"""
    try:
        # Clova 모델 사용 (기존 로직 유지)
        clova_model = ModelSingleton.get_instance()
        result = clova_model.classify(text=message)
        is_toxic = any(
            keyword in result
            for keyword in ["유해합니다", "제재 대상", "부적절", "비속어"]
        )

        return ChatReportResponse(message=message, isToxic=is_toxic)
    except Exception as e:
        logger.error(f"Chat report processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@log_performance(operation_name="handle_chat_report", include_memory=True)
async def handle_chat_reports(body: ChatReportRequest) -> ChatReportResponse:
    """완전한 채팅 신고 처리 (룰베이스 + AI 모델 + DB 저장)"""
    try:
        # 유해성 분석
        toxicity_result = await analyze_toxicity(body.messageContent)

        # 데이터베이스 저장
        report_data = create_report_document(
            body.messageId, body.messageContent, toxicity_result
        )
        save_report_to_db(report_data)

        # 응답 생성
        return create_response(toxicity_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat report processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
