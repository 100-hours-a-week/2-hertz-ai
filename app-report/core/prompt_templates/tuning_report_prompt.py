# 뉴스 생성 프롬프트 템플릿

from ...schemas.tuning_schema import UserProfile


def build_prompt(category: str, userA: UserProfile, userB: UserProfile) -> str:
    def format_list(items: list[str]) -> str:
        return ", ".join(items) if items else "없음"

    return (
        f"[매칭 유형: {category}]\n\n"
        f"🔹 사용자 A\n"
        f" - MBTI: {userA.MBTI}\n"
        f" - 종교: {userA.religion}\n"
        f" - 흡연: {userA.smoking}\n"
        f" - 음주: {userA.drinking}\n"
        f" - 성향: {format_list(userA.personality)}\n"
        f" - 선호하는 사람 유형: {format_list(userA.preferredPeople)}\n"
        f" - 관심사: {format_list(userA.currentInterests)}\n"
        f" - 좋아하는 음식: {format_list(userA.favoriteFoods)}\n"
        f" - 좋아하는 운동: {format_list(userA.likedSports)}\n"
        f" - 반려동물: {format_list(userA.pets)}\n"
        f" - 자기계발 활동: {format_list(userA.selfDevelopment)}\n"
        f" - 취미: {format_list(userA.hobbies)}\n\n"
        f"🔹 사용자 B\n"
        f" - MBTI: {userB.MBTI}\n"
        f" - 종교: {userB.religion}\n"
        f" - 흡연: {userB.smoking}\n"
        f" - 음주: {userB.drinking}\n"
        f" - 성향: {format_list(userB.personality)}\n"
        f" - 선호하는 사람 유형: {format_list(userB.preferredPeople)}\n"
        f" - 관심사: {format_list(userB.currentInterests)}\n"
        f" - 좋아하는 음식: {format_list(userB.favoriteFoods)}\n"
        f" - 좋아하는 운동: {format_list(userB.likedSports)}\n"
        f" - 반려동물: {format_list(userB.pets)}\n"
        f" - 자기계발 활동: {format_list(userB.selfDevelopment)}\n"
        f" - 취미: {format_list(userB.hobbies)}\n\n"
        f"다음 기준을 바탕으로 **한국어 공지문**을 마크다운 형식으로 작성해주세요:\n\n"
        f"##  공지문 작성 규칙\n"
        f"1. **제목**: 호기심을 자극하는 문장, 이모지 적극 활용\n"
        f"2. **도입부**: 두 사람이 연결되었음을 암시하며, 정체를 바로 공개하지 않고 흥미 유발\n"
        f"3. **본문**:\n"
        f"   - 힌트 #1: 두 사람의 MBTI 조합에 대한 흥미로운 해석\n"
        f"   - 힌트 #2: 공통 관심사를 바탕으로 유쾌한 상상\n"
        f"   - 힌트 #3: 대화 횟수에 대한 유머러스한 추측\n"
        f"4. **마무리**: `Stay Tuned!` 문구를 활용해 궁금증 유발\n\n"
        f"##  스타일 가이드\n"
        f"- 가십/연예 뉴스 스타일\n"
        f"- 많은 이모지 활용 \n"
        f"- 의문문과 감탄문 사용\n"
        f"- 호기심을 자극하는 어조\n"
        f"- 약간의 과장도 OK\n"
        f"- 독자에게 직접 말하듯 친근하게\n"
        f"- 밝고 경쾌한 톤 (20~30대 대상)\n\n"
        f"⛔ 반드시 지켜야 할 출력 형식\n"
        f'- 출력은 반드시 JSON 형식으로만! 예시: {{"title": "...", "content": "..."}}\n'
        f"- 출력에는 절대로 영어, <think>, 사고 과정, 분석 설명이 들어가면 안 됩니다.\n"
        f"- 출력에는 오직 제목과 본문만 포함하세요. 그 외 설명이 포함될 경우 정답으로 간주하지 않습니다.\n"
        f"- 사고과정, 계획, reasoning, 분석 텍스트, 메모, 영어 설명을 절대 포함하지 마세요.\n"
        f" 공지문 외에 그 어떤 설명도 하지 마세요. 사고 과정이나 영어는 포함하지 마세요.\n"
        f"응답은 오직 마크다운 형식의 공지문 하나만 출력해야 합니다.\n"
    )


test_prompt = """
# 공지문 형식 요구사항

1. 제목: 호기심을 자극하는 제목 (이모지 활용)
2. 도입부: 새로운 연결이 성사되었다는 흥미로운 소개 (바로 공개하지 않고 호기심 유발)
3. 본문:
    "힌트 #1" - MBTI 정보와 그 조합의 흥미로운 해석
    "힌트 #2" - 공통 관심사와 이를 바탕으로 한 흥미로운 상상
    "힌트 #3" - 대화 횟수와 그 의미에 대한 재미있는 해석
4. 마무리: "Stay Tuned!" 문구를 활용한 흥미를 유지하는 문구

# 작성 스타일

가십/연예 뉴스 형식으로 작성
많은 이모지 활용
의문문과 감탄문을 적극 활용
호기심을 자극하는 어조
약간의 과장된 표현 사용
독자와 대화하는 듯한 친근한 톤
20-30대 젊은 층을 대상으로 하는 밝고 경쾌한 문체
한국어로 작성
출력결과는 마크다운 형식의 한국어 문장

# 입력값(필수 입력 정보)
MBTI: INTJ, ENFP
공통관심사: 디지털 드로잉, 넷플릭스 심야 감상
대화횟수: 17회
"""
