# app/services/tuning_report_service.py

import json
import logging
import re
from typing import Dict, Union

from core.prompt_templates.tuning_report_prompt import build_prompt
from models import qwen_loader
from schemas.tuning_schema import TuningReport, TuningReportResponse

logger = logging.getLogger(__name__)


def clean_and_extract_response(content: str) -> Dict[str, str]:
    """
    LLM 응답에서 불필요한 메타 텍스트 제거 및 JSON 파싱 시도

    Args:
        content: LLM 모델에서 생성된 원본 텍스트

    Returns:
        파싱된 title과 content를 포함한 딕셔너리
    """
    # 응답에서 코드 블록이나 시스템 메시지 제거
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"```$", "", content)
    content = content.strip()

    # JSON 파싱 시도
    try:
        result = json.loads(content)
        if isinstance(result, dict) and "title" in result and "content" in result:
            return result
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse response as JSON: {content[:100]}...")

    # JSON 파싱 실패 시 정규식으로 제목과 내용 추출 시도
    title_match = re.search(r"#\s*(.+?)\n", content)
    title = (
        title_match.group(1).strip() if title_match else "📢 [속보] 새로운 튜닝 연결!"
    )

    # 제목 라인을 내용에서 제거
    if title_match:
        content = content.replace(title_match.group(0), "", 1)

    return {"title": title, "content": content.strip()}


async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    """
    튜닝 리포트를 생성하는 서비스 함수

    Args:
        request: 튜닝 리포트 생성 요청 데이터

    Returns:
        생성된 튜닝 리포트 응답

    Raises:
        Exception: 리포트 생성 중 오류 발생 시
    """
    try:
        # 프롬프트 생성
        prompt_text = build_prompt(
            category=request.category, userA=request.userA, userB=request.userB
        )

        # 모델 요청 메시지 구성
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 데이팅 앱의 작가입니다. 주어진 사용자 정보로 흥미롭고 유쾌한 **공지문 제목과 본문만** 작성해주세요. "
                    "기사 외에는 어떤 설명이나 생각도 쓰지 마세요. 기사 내용은 마크다운 형식으로 작성하고 "
                    "기사를 json 형식으로 아래처럼 구성하세요:\n\n"
                    '{ "title": "기사 제목", "content": "기사 내용" }'
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        # 모델 인스턴스 가져오기
        model = qwen_loader.get_model()

        # 응답 생성
        logger.info("Generating tuning report...")
        model_response = model.get_response(messages)

        # 응답 상태 코드 확인
        if model_response.get("status_code") != 200:
            error_msg = model_response.get("error", "Unknown error")
            logger.error(f"Model inference failed: {error_msg}")
            raise Exception(f"Failed to generate report: {error_msg}")

        # 응답 본문 전처리 및 파싱
        raw_content = model_response.get("content", "")
        logger.debug(f"Raw model response: {raw_content[:200]}...")

        parsed = clean_and_extract_response(raw_content)

        title = parsed.get("title", "")
        content = parsed.get("content", "")

        logger.info(f"Generated report with title: {title[:50]}...")

        # 결과 반환
        return TuningReportResponse(
            code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
        )

    except Exception as e:
        logger.error(f"Error generating tuning report: {str(e)}")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        raise e
