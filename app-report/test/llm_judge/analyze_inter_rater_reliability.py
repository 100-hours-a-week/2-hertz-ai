from scipy.stats import pearsonr

# 샘플 데이터
data = {
    # khloe
    "evaluator1": {
        "headline_attention": [
            3,
            1,
            2,
            3,
            3,
            5,
            4,
            4,
            5,
            4,
            2,
            4,
            2,
            3,
            2,
            2,
            5,
            4,
            4.5,
            4,
        ],
        "content_interest": [
            1,
            2,
            2,
            2,
            3,
            5,
            4,
            4,
            3,
            3,
            2,
            3,
            3,
            2,
            2,
            3,
            5,
            5,
            5,
            4.5,
        ],
        "content_readability": [
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2.5,
            2,
            2,
            2,
            2,
            1,
            2.8,
            2.8,
            3,
            3,
        ],
        "concept_completeness": [
            2,
            2,
            1,
            1,
            1,
            2,
            2.5,
            2.5,
            2,
            1.5,
            1.5,
            2,
            2,
            2,
            1,
            2.5,
            3,
            3,
            3,
            3,
        ],
    },
    # noah
    "evaluator2": {
        "headline_attention": [
            3,
            2,
            2,
            3,
            3,
            4,
            5,
            4,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            3,
            5,
            4,
        ],
        "content_interest": [
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            4,
            3,
            5,
            5,
        ],
        "content_readability": [
            2,
            2,
            2,
            2,
            2,
            3,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
        ],
        "concept_completeness": [
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            3,
            3,
            3,
            3,
            3,
        ],
    },
}


def calculate_pearson_correlation(data):
    """
    평가자 간 피어슨 상관계수를 계산하는 함수

    Args:
        data: 평가자별 점수 데이터가 담긴 딕셔너리

    Returns:
        correlations: 항목별 상관계수 결과
        total_correlation: 총합 점수 상관계수
    """

    # 결과 저장용 딕셔너리
    correlations = {}

    # 각 평가항목별 상관계수 계산
    evaluation_items = [
        "headline_attention",
        "content_interest",
        "content_readability",
        "concept_completeness",
    ]

    print("=" * 60)
    print("평가자 간 피어슨 상관계수 분석 결과")
    print("=" * 60)

    for item in evaluation_items:
        evaluator1_scores = data["evaluator1"][item]
        evaluator2_scores = data["evaluator2"][item]

        # 피어슨 상관계수 계산
        correlation, p_value = pearsonr(evaluator1_scores, evaluator2_scores)
        correlations[item] = {
            "correlation": correlation,
            "p_value": p_value,
            "significance": "significant" if p_value < 0.05 else "not significant",
        }

        # 한국어 항목명 변환
        korean_names = {
            "headline_attention": "헤드라인 주목도",
            "content_interest": "본문 재미",
            "content_readability": "본문 가독성",
            "concept_completeness": "컨셉 완성도",
        }

        print(f"\n{korean_names[item]}:")
        print(f"  피어슨 상관계수: {correlation:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  통계적 유의성: {correlations[item]['significance']}")

        # 상관계수 해석
        if abs(correlation) >= 0.8:
            interpretation = "매우 높은 상관관계"
        elif abs(correlation) >= 0.6:
            interpretation = "높은 상관관계"
        elif abs(correlation) >= 0.4:
            interpretation = "중간 상관관계"
        elif abs(correlation) >= 0.2:
            interpretation = "낮은 상관관계"
        else:
            interpretation = "거의 상관관계 없음"

        print(f"  해석: {interpretation}")

    # 총합 점수 계산 및 상관계수 분석
    evaluator1_total = []
    evaluator2_total = []

    for i in range(len(data["evaluator1"]["headline_attention"])):
        total1 = (
            data["evaluator1"]["headline_attention"][i]
            + data["evaluator1"]["content_interest"][i]
            + data["evaluator1"]["content_readability"][i]
            + data["evaluator1"]["concept_completeness"][i]
        )

        total2 = (
            data["evaluator2"]["headline_attention"][i]
            + data["evaluator2"]["content_interest"][i]
            + data["evaluator2"]["content_readability"][i]
            + data["evaluator2"]["concept_completeness"][i]
        )

        evaluator1_total.append(total1)
        evaluator2_total.append(total2)

    total_correlation, total_p_value = pearsonr(evaluator1_total, evaluator2_total)

    print(f"\n총합 점수:")
    print(f"  피어슨 상관계수: {total_correlation:.4f}")
    print(f"  p-value: {total_p_value:.4f}")
    print(
        f"  통계적 유의성: {'significant' if total_p_value < 0.05 else 'not significant'}"
    )

    if abs(total_correlation) >= 0.8:
        interpretation = "매우 높은 상관관계"
    elif abs(total_correlation) >= 0.6:
        interpretation = "높은 상관관계"
    elif abs(total_correlation) >= 0.4:
        interpretation = "중간 상관관계"
    elif abs(total_correlation) >= 0.2:
        interpretation = "낮은 상관관계"
    else:
        interpretation = "거의 상관관계 없음"

    print(f"  해석: {interpretation}")

    return correlations, {
        "correlation": total_correlation,
        "p_value": total_p_value,
        "evaluator1_total": evaluator1_total,
        "evaluator2_total": evaluator2_total,
    }


# 실행 예시
if __name__ == "__main__":
    # 상관계수 계산
    correlations, total_correlation = calculate_pearson_correlation(data)
