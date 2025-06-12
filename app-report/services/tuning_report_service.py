# 뉴스 콘텐츠 생성 서비스(로컬 테스트용)
import json
import re

from core.enum_process import convert_to_korean
from core.prompt_templates.tuning_report_prompt import build_prompt
from models import qwen_loader
from schemas.tuning_schema import TuningReport, TuningReportResponse, UserProfile
from utils.logger import log_performance, logger


def clean_and_extract_response(content: str) -> dict:
    """
    LLM 응답에서 불필요한 메타 텍스트 제거 및 JSON 파싱 시도
    다양한 형태의 JSON 응답을 처리
    """
    # 1. 기본 전처리
    clean_text = re.sub(r"^<[^>]+>\s*", "", content).strip()

    # 2. JSON 블록 추출 시도 (```json ... ``` 형태)
    json_block_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL
    )
    if json_block_match:
        clean_text = json_block_match.group(1)

    # 3. 첫 번째 JSON 객체 추출 시도
    json_match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(0)

    try:
        # 4. 첫 번째 파싱 시도
        result = json.loads(clean_text)

        # 5. 중첩된 JSON 문자열 처리
        if isinstance(result.get("content"), str):
            content_str = result["content"]

            # content가 JSON 문자열인지 확인
            if content_str.strip().startswith("{") and content_str.strip().endswith(
                "}"
            ):
                try:
                    # JSON 문자열을 파싱
                    inner_json = json.loads(content_str)
                    if (
                        isinstance(inner_json, dict)
                        and "title" in inner_json
                        and "content" in inner_json
                    ):
                        return inner_json
                except json.JSONDecodeError:
                    pass

            # content에서 JSON 패턴 추출 시도
            inner_json_match = re.search(
                r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', content_str, re.DOTALL
            )
            if inner_json_match:
                try:
                    inner_json = json.loads(inner_json_match.group(0))
                    if (
                        isinstance(inner_json, dict)
                        and "title" in inner_json
                        and "content" in inner_json
                    ):
                        return inner_json
                except json.JSONDecodeError:
                    pass

        # 6. 정상적인 JSON 구조인 경우 그대로 반환
        if "title" in result and "content" in result:
            return result

        # 7. 다른 구조의 JSON인 경우 변환 시도
        return {
            "title": result.get("title", ""),
            "content": str(result.get("content", clean_text)),
        }

    except json.JSONDecodeError as e:
        # 8. JSON 파싱 실패 시 정규식으로 title과 content 추출 시도
        title_match = re.search(
            r'"title"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', clean_text, re.DOTALL
        )
        content_match = re.search(
            r'"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', clean_text, re.DOTALL
        )

        if title_match and content_match:
            # 이스케이프 문자 처리
            title = (
                title_match.group(1)
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
            )
            content = (
                content_match.group(1)
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
            )
            return {"title": title, "content": content}

        # 9. 모든 파싱 시도 실패 시 기본값 반환
        logger.warning(f"JSON 파싱 실패: {e}")
        return {"title": "", "content": clean_text}


@log_performance(
    operation_name="generate_tuning_report", include_memory=True, include_args=True
)
async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    """
    소셜봇 채팅 메시지를 생성하는 서비스
    """
    try:
        logger.info(
            f"프롬프트 생성 시작 [category={request.category}, chatCount={request.chatCount}]"
        )
        userA = UserProfile(**convert_to_korean(request.userA.model_dump()))
        userB = UserProfile(**convert_to_korean(request.userB.model_dump()))

        prompt_text = build_prompt(
            category=request.category,
            chatCount=request.chatCount,
            userA=userA,
            userB=userB,
        )
        logger.debug(
            f"생성된 프롬프트: {prompt_text[:30]}..."
        )  # 과도한 로그 방지를 위해 일부만 출력

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 소셜 디스커버리 앱 '튜닝'의 유쾌한 감성 기자입니다.\n"
                    "당신의 임무는 주어진 두 사람의 정보를 바탕으로만 **흥미롭고 감정선이 살아있는 공지 기사**를 작성하는 것입니다.\n"
                    "\n"
                    "⛔ 반드시 지켜야 할 규칙:\n"
                    "1. 응답은 반드시 JSON 한 개만 반환 (문자열 아님, 마크다운 아님, 설명/코드블럭 사용 금지)\n"
                    "2. 출력 형식 예시:\n"
                    '{ "title": "기사 제목", "content": "공지문 본문 내용" }\n'
                    "\n"
                    "✅ 공지문 작성 스타일:\n"
                    "- 문단마다 자연스러운 이야기 흐름이 있도록 연결하세요.\n"
                    "- 감정, 대화, 상상력을 구체적으로 드러내세요.\n"
                    "- 각 문단은 2~3문장이며, 이모지는 자연스럽게 넣되 과도하게 사용하지 마세요.\n"
                    "- FRIEND 유형에서는 연애나 썸 관련 표현 금지. 오직 우정/의리 중심으로 작성하세요.\n"
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        logger.info("▶ 모델 호출 시작")
        model = qwen_loader.get_model()
        model_response = model.invoke(messages)
        logger.info("▶ 모델 응답 수신 완료")

        # 응답 본문 전처리 및 파싱
        raw_content = model_response.content
        parsed = clean_and_extract_response(raw_content)

        title = parsed.get("title", "")
        content = parsed.get("content", "")
        logger.info(f"[SUCCESS] 튜닝 리포트 생성 완료 | title={title}")

        return TuningReportResponse(
            code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
        )

    except Exception as e:
        logger.exception("[FAIL] 튜닝 리포트 생성 중 예외 발생")
        raise e
