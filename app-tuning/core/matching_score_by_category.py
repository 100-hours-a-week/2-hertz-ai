"""
매칭 스코어 계산 모듈
사용자 간 유사도를 계산하기 위한 다양한 알고리즘과 유틸리티 함수 제공

주요 기능:
1. 임베딩 벡터 처리 및 평균화
2. MBTI 호환성 계산
3. 연령대 일치도 계산
4. 관심사/태그 매칭 계산
5. 규칙 기반 유사도 계산
6. 최종 매칭 점수 통합 계산
"""

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from core.embedding import user_data_to_sentence
from models.sbert_loader import get_model
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import log_performance

# ---------------------- 상수 정의 ----------------------
# 모델 임베딩 차원 및 가중치 상수
EMBEDDING_DIM = 768  # SBERT 모델의 임베딩 차원

# MBTI 관련 상수
MBTI_WEIGHTS = [0.5, 1.0, 1.0, 0.5]  # E/I, N/S, F/T, J/P 각 차원별 가중치

# 임베딩 계산에 사용할 필드 목록
EMBEDDING_FIELDS = [
    "currentInterests",
    "favoriteFoods",
    "likedSports",
    "pets",
    "selfDevelopment",
    "hobbies",
]

# MBTI 유형 간 호환성 맵핑
# 각 MBTI 유형에 대해 잘 맞는 상호보완적 유형 목록 정의
MBTI_COMPATIBILITY = {
    "INTJ": ["ENFP", "ENTP"],
    "INTP": ["ENTJ", "ENFJ"],
    "INFJ": ["ENFP", "ENTP"],
    "INFP": ["ENFJ", "ESFJ"],
    "ISTJ": ["ESFP", "ESTP"],
    "ISTP": ["ESFJ", "ENFJ"],
    "ISFJ": ["ESTP", "ESFP"],
    "ISFP": ["ENFJ", "ESFJ"],
    "ENTJ": ["INFP", "INTP"],
    "ENTP": ["INFJ", "INTJ"],
    "ENFJ": ["INFP", "ISFP"],
    "ENFP": ["INFJ", "INTJ"],
    "ESTJ": ["ISFP", "ISTP"],
    "ESTP": ["ISFJ", "ISTJ"],
    "ESFJ": ["ISFP", "INFP"],
    "ESFP": ["ISFJ", "ISTJ"],
}

# 연령대 그룹 정의 및 순서 설정
AGE_GROUPS = {
    "AGE_10S": 1,
    "AGE_20S": 2,
    "AGE_30S": 3,
    "AGE_40S": 4,
    "AGE_50S": 5,
    "AGE_60S": 6,
}


def average_field_embedding(field_embeddings: dict, fields: list) -> list:
    """
    개별 필드 임베딩들을 평균하여 하나의 통합 벡터로 만드는 함수

    Args:
        field_embeddings: 필드별 임베딩 벡터 딕셔너리
        fields: 평균 계산에 사용할 필드 이름 목록

    Returns:
        평균 임베딩 벡터 (리스트)
    """
    # 유효한 임베딩 벡터만 추출
    vectors = [field_embeddings.get(f) for f in fields if f in field_embeddings]
    vectors = [v for v in vectors if v is not None]

    # 벡터가 없을 경우 기본 영벡터 반환
    if not vectors:
        return [0.0] * EMBEDDING_DIM

    # 평균 계산 후 리스트로 변환하여 반환
    return np.mean(np.array(vectors), axis=0).tolist()


def mbti_weighted_score(
    mbti1: str, mbti2: str, weight_similarity: float = 0.7
) -> float:
    """
    두 MBTI 유형 간의 호환성 점수 계산
    각 MBTI 차원에 가중치를 적용하고, 궁합 정보도 고려하여 종합 점수 산출

    Args:
        mbti1: 첫 번째 사용자의 MBTI
        mbti2: 두 번째 사용자의 MBTI
        weight_similarity: 유사성 점수와 호환성 점수 간의 가중치 (기본값: 0.7)

    Returns:
        0.0~1.0 사이의 호환성 점수
    """
    # MBTI 유효성 검사 (빈 값, 알 수 없음, 형식 불일치 등)
    is_invalid1 = not mbti1 or mbti1 not in MBTI_COMPATIBILITY or len(mbti1) != 4
    is_invalid2 = not mbti2 or mbti2 not in MBTI_COMPATIBILITY or len(mbti2) != 4

    # 양쪽 다 유효하지 않은 경우
    if is_invalid1 and is_invalid2:
        return 0.5  # 중간값 반환 (둘 다 없음 → 사용자 고립 & 패널티 부여 방지)
    # 한쪽만 유효하지 않은 경우
    elif is_invalid1 or is_invalid2:
        return 0.6  # 정보 불균형 상태지만, 매칭 기회는 열어둠

    # 차원별 일치 유사성 점수 계산 (0~1)
    # 각 차원(E/I, N/S, F/T, J/P)마다 일치하면 해당 가중치 적용
    similarity_score = sum(
        MBTI_WEIGHTS[i] for i in range(4) if mbti1[i] == mbti2[i]
    ) / sum(MBTI_WEIGHTS)

    # 상호보완적 호환성 점수 계산 (0 또는 1)
    # 정의된 상호보완적 유형 목록에 포함되면 최대 점수
    compatibility_score = 1.0 if mbti2 in MBTI_COMPATIBILITY.get(mbti1, []) else 0.0

    # 가중치를 적용한 최종 점수 계산
    final_score = (weight_similarity * similarity_score) + (
        (1 - weight_similarity) * compatibility_score
    )

    return round(final_score, 6)


def age_group_match_score(a: str, b: str) -> float:
    """
    두 연령대 간의 일치도 점수 계산

    Args:
        a: 첫 번째 사용자의 연령대
        b: 두 번째 사용자의 연령대

    Returns:
        연령대 일치도 점수:
        - 완전 일치: 1.0
        - 한 단계 차이(예: 20대-30대): 0.5
        - 두 단계 이상 차이: 0.0
    """
    # 연령대 코드를 숫자 값으로 변환
    a_val = AGE_GROUPS.get(a)
    b_val = AGE_GROUPS.get(b)

    # 유효하지 않은 연령대 코드인 경우
    if a_val is None or b_val is None:
        return 0.0

    # 연령대 차이 계산
    diff = abs(a_val - b_val)

    # 차이에 따른 점수 반환
    if diff == 0:  # 동일 연령대
        return 1.0
    elif diff == 1:  # 인접 연령대 (예: 20대-30대)
        return 0.5
    else:  # 2단계 이상 차이
        return 0.0


def match_tags(list1: List[str], list2: List[str]) -> float:
    """
    두 태그 목록 간의 유사도 계산 (자카드 유사도)

    Args:
        list1: 첫 번째 태그 목록
        list2: 두 번째 태그 목록

    Returns:
        0.0~1.0 사이의 유사도 점수 (공통 태그 수 / 전체 고유 태그 수)
    """
    # 빈 목록 처리
    if not list1 or not list2:
        return 0.0

    # 교집합과 합집합 계산
    overlap = set(list1) & set(list2)
    union = set(list1) | set(list2)

    # 자카드 유사도 계산 및 반환
    return round(len(overlap) / len(union), 6)


def rule_based_similarity(user1: dict, user2: dict) -> float:
    """
    사용자 프로필 데이터를 기반으로 규칙 기반 유사도 점수 계산
    여러 속성(종교, 흡연, 음주, MBTI, 연령대, 성격 등)의 일치도를 종합

    Args:
        user1: 첫 번째 사용자 프로필 데이터
        user2: 두 번째 사용자 프로필 데이터

    Returns:
        0.0~1.0 사이의 규칙 기반 유사도 점수
    """
    # 기본 필드 일치도 (종교, 흡연, 음주 등) - 단순 일치 여부 확인
    base_fields = ["religion", "smoking", "drinking"]
    base_score = sum(1 for f in base_fields if user1.get(f) == user2.get(f)) / len(
        base_fields
    )

    # MBTI 호환성 점수
    mbti_score = mbti_weighted_score(user1.get("MBTI"), user2.get("MBTI"))

    # 연령대 일치도 점수
    age_score = age_group_match_score(user1.get("ageGroup"), user2.get("ageGroup"))

    # 선호-성격 매칭 점수 (양방향)
    # user1의 선호 특성과 user2의 성격 특성 비교
    pref_score = match_tags(
        user1.get("preferredPeople", []), user2.get("personality", [])
    )
    # user2의 선호 특성과 user1의 성격 특성 비교
    rev_pref_score = match_tags(
        user2.get("preferredPeople", []), user1.get("personality", [])
    )

    # 가중치를 적용한 최종 점수 계산
    # - 기본 필드(종교,흡연,음주): 30%
    # - MBTI 호환성: 20%
    # - 연령대 일치도: 20%
    # - 선호-성격 매칭(양방향 평균): 30% -> 삭제

    final_score = (
        base_score * 0.3
        + mbti_score * 0.2
        + age_score * 0.2
        + (pref_score + rev_pref_score) / 2 * 0.3
    )

    # 소수점 6자리로 반올림하여 반환
    return round(final_score, 6)


def rule_based_similarity_v3(user1: dict, user2: dict) -> float:
    """
    사용자 프로필 데이터를 기반으로 규칙 기반 유사도 점수 계산
    여러 속성(MBTI, 연령대, 성격 등)의 일치도를 종합

    Args:
        user1: 첫 번째 사용자 프로필 데이터
        user2: 두 번째 사용자 프로필 데이터

    Returns:
        0.0~1.0 사이의 규칙 기반 유사도 점수
    """
    # MBTI 호환성 점수
    mbti_score = mbti_weighted_score(user1.get("MBTI"), user2.get("MBTI"))

    # 연령대 일치도 점수
    age_score = age_group_match_score(user1.get("ageGroup"), user2.get("ageGroup"))

    # 가중치를 적용한 최종 점수 계산
    # - MBTI 호환성: 50%
    # - 연령대 일치도: 50%

    final_score = mbti_score * 0.5 + age_score * 0.5

    # 소수점 6자리로 반올림하여 반환
    return round(final_score, 6)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    벡터를 L2 정규화하여 단위 벡터로 변환

    Args:
        vector: 정규화할 벡터

    Returns:
        정규화된 단위 벡터
    """
    norm = np.linalg.norm(vector)
    if norm == 0:  # 영벡터인 경우 처리
        return vector
    return vector / norm


# combine_embeddings, average_field_embedding 등 임베딩 결합 함수는 더이상 사용하지 않으므로 아래와 같이 주석 처리 또는 삭제합니다.
def combine_embeddings(
    profile_embedding: List[float],
    field_embeddings: dict,
) -> np.ndarray:
    """
    프로필 임베딩과 필드별 임베딩을 가중 평균으로 결합

    Args:
       profile_embedding: 프로필 임베딩 벡터
       field_embeddings: 필드별 임베딩 사전

    Returns:
        결합된 임베딩 벡터
    """
    # 프로필 임베딩을 numpy 배열로 변환
    profile_embed = np.array(profile_embedding)

    # 필드 임베딩 추출 및 평균 계산
    avg_field_embed = np.array(
        average_field_embedding(field_embeddings, EMBEDDING_FIELDS)
    )

    # 정규화된 벡터의 가중 평균
    norm_profile = normalize_vector(profile_embed)
    norm_fields = normalize_vector(avg_field_embed)

    # 프로필(60%)과 필드 임베딩(40%)의 가중 평균
    return 0.6 * norm_profile + 0.4 * norm_fields


@log_performance(operation_name="compute_matching_score", include_memory=True)
def compute_matching_score(
    user_id: str,
    user_embedding: List[float],
    user_meta: dict,
    all_users: dict,
    category: str,
) -> Dict[str, float]:
    """
    최적화된 매칭 점수 계산 함수
    벡터화 및 배치 처리를 통해 성능 개선

    Args:
        user_id: 기준 사용자 ID
        user_embedding: 기준 사용자의 임베딩 벡터
        user_meta: 기준 사용자의 메타데이터
        all_users: 전체 사용자 데이터 (IDs, 임베딩, 메타데이터)
        embedding_method: 임베딩 결합 방식

    Returns:
        사용자 ID를 키로, 매칭 점수를 값으로 하는 딕셔너리
    """
    # 1. 가중치 정의
    weights_by_category = {
        "friend": {"embedding": 0.7, "rule": 0.3},
        "couple": {"embedding": 0.6, "rule": 0.4},
    }
    weights = weights_by_category.get(category, weights_by_category[category])
    embedding_weight = weights["embedding"]
    rule_weight = weights["rule"]

    # 2. 사용자 도메인 기준 필터링
    all_ids = all_users["ids"]
    # all_embeddings = all_users["embeddings"]
    all_metas = all_users["metadatas"]
    domain = user_meta.get("emailDomain")
    my_gender = user_meta.get("gender")

    # 도메인 필터링 먼저 수행 (동일 도메인 사용자만 선택)
    domain_indices = []
    for i, meta in enumerate(all_metas):
        if all_ids[i] == user_id:
            continue
        if meta.get("emailDomain") != domain:
            continue
        if category == "couple":
            # 커플 카테고리: 성별 반대만
            if my_gender and meta.get("gender") == my_gender:
                continue
        domain_indices.append(i)

    if not domain_indices:
        return {}

    # 3. 나의 임베딩 + 필드 결합
    my_fields = json.loads(user_meta.get("field_embeddings", "{}"))
    combined_user_embedding = combine_embeddings(user_embedding, my_fields)

    other_ids, other_embeddings, other_metas_filtered = [], [], []
    for i in domain_indices:
        other_meta = all_metas[i]
        other_fields = json.loads(other_meta.get("field_embeddings", "{}"))
        combined_other_embedding = np.array(
            average_field_embedding(other_fields, EMBEDDING_FIELDS)
        )

        other_ids.append(all_ids[i])
        other_embeddings.append(combined_other_embedding)
        other_metas_filtered.append(other_meta)

    other_embeddings_matrix = np.vstack(other_embeddings)
    cosine_sims = cosine_similarity([combined_user_embedding], other_embeddings_matrix)[
        0
    ]

    # 4. 최종 유사도 계산
    similarities = {}
    for idx, other_id in enumerate(other_ids):
        rule_sim = rule_based_similarity(user_meta, other_metas_filtered[idx])
        final_score = embedding_weight * cosine_sims[idx] + rule_weight * rule_sim
        similarities[other_id] = round(final_score, 6)

    return similarities


def compute_matching_score_sentence_based(
    user_id: str,
    user_meta: dict,
    all_users: dict,
    category: str,
) -> dict:
    """
    문장 임베딩 기반 매칭 점수 계산 (유저 메타데이터를 한국어 문장으로 변환 후 임베딩)
    """

    # 1. 가중치 정의
    weights_by_category = {
        "friend": {"embedding": 0.7, "rule": 0.3},
        "couple": {"embedding": 0.6, "rule": 0.4},
    }
    weights = weights_by_category.get(category, weights_by_category["friend"])
    embedding_weight = weights["embedding"]
    rule_weight = weights["rule"]

    # 2. 데이터를 Pandas DataFrame으로 변환 (한 번만 수행)
    df = pd.DataFrame(all_users["metadatas"])
    df["id"] = all_users["ids"]

    # 3. Pandas를 이용한 초고속 필터링
    domain = user_meta.get("emailDomain")  # noqa: F841
    my_gender = user_meta.get("gender")  # noqa: F841

    # 기본 필터링 조건
    query = "(id != @user_id) & (emailDomain == @domain)"
    if category == "couple" and my_gender:
        query += " & (gender != @my_gender)"

    filtered_df = df.query(query).reset_index()  # 쿼리를 통해 조건에 맞는 사용자만 선택

    if filtered_df.empty:
        return {}

    # 4. 임베딩 계산 (필터링된 사용자에 대해서만 수행)
    model = get_model()

    my_text = user_data_to_sentence(user_meta)
    my_embedding = model.encode(my_text, show_progress_bar=False)

    other_texts = filtered_df.apply(user_data_to_sentence, axis=1).tolist()
    other_embeddings_matrix = model.encode(other_texts, show_progress_bar=False)

    # 5. 유사도 및 점수 계산 (벡터화 연산)
    cosine_sims = cosine_similarity([my_embedding], other_embeddings_matrix)[0]

    # 규칙 기반 유사도 계산
    # rule_sims = calculate_rule_scores_in_batch(user_meta, filtered_df)

    rule_sims = np.array(
        [
            rule_based_similarity_v3(user_meta, row.to_dict())
            for _, row in filtered_df.iterrows()
        ]
    )

    final_scores = embedding_weight * cosine_sims + rule_weight * rule_sims

    # 6. 최종 결과 생성
    similarities = {
        other_id: round(score, 6)
        for other_id, score in zip(filtered_df["id"], final_scores)
    }

    return similarities
