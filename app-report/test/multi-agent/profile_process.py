from typing import List, Set

from ..schemas.tuning_schema import UserProfile


def extract_all_interests(user: UserProfile) -> Set[str]:
    """사용자의 모든 관심사를 추출 (정규화 적용)"""
    all_interests = set()

    # 각 필드에서 관심사 추출
    fields_to_check = [
        user.currentInterests,  # 관심사
        user.favoriteFoods,  # 좋아하는 음식
        user.likedSports,  # 좋아하는 운동
        user.pets,  # 반려동물
        user.selfDevelopment,  # 자기계발 활동
        user.hobbies,  # 취미
    ]

    for field in fields_to_check:
        if field:  # None이 아닌 경우
            for item in field:
                if item and item.strip():  # 빈 문자열이 아닌 경우
                    all_interests.add(item)

    return all_interests


def find_exact_matches(interests_a: Set[str], interests_b: Set[str]) -> List[str]:
    """정확히 일치하는 관심사 찾기"""
    return list(interests_a.intersection(interests_b))
