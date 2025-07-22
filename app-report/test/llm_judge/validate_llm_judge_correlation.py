from scipy.stats import pearsonr


def calculate_human_average_and_store(human_data):
    """두 평가자의 점수를 평균내어 human_evaluator에 저장"""

    evaluation_items = [
        "headline_attention",
        "content_interest",
        "content_readability",
        "concept_completeness",
    ]

    # human_evaluator 키 추가
    data = {}
    data["human_evaluator"] = {}

    for item in evaluation_items:
        scores1 = human_data["evaluator1"][item]
        scores2 = human_data["evaluator2"][item]

        # 두 평가자의 평균 계산
        avg_scores = []
        for s1, s2 in zip(scores1, scores2):
            if s1 is not None and s2 is not None:
                avg_scores.append((s1 + s2) / 2)
            else:
                avg_scores.append(None)

        # human_evaluator에 저장
        data["human_evaluator"][item] = avg_scores

    #    print(f"{item}: {avg_scores}")

    return data


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
    print("평가자-LLM 간 피어슨 상관계수 분석 결과")
    print("=" * 60)

    for item in evaluation_items:
        human_evaluator_scores = data["human_evaluator"][item]
        llm_evaluator_scores = data["llm_evaluator"][item]

        # 피어슨 상관계수 계산
        correlation, p_value = pearsonr(human_evaluator_scores, llm_evaluator_scores)
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
    human_evaluator_total = []
    llm_evaluator_total = []

    for i in range(len(data["human_evaluator"]["headline_attention"])):
        total1 = (
            data["human_evaluator"]["headline_attention"][i]
            + data["human_evaluator"]["content_interest"][i]
            + data["human_evaluator"]["content_readability"][i]
            + data["human_evaluator"]["concept_completeness"][i]
        )

        total2 = (
            data["llm_evaluator"]["headline_attention"][i]
            + data["llm_evaluator"]["content_interest"][i]
            + data["llm_evaluator"]["content_readability"][i]
            + data["llm_evaluator"]["concept_completeness"][i]
        )

        human_evaluator_total.append(total1)
        llm_evaluator_total.append(total2)

    total_correlation, total_p_value = pearsonr(
        human_evaluator_total, llm_evaluator_total
    )

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
        "human_evaluator_total": human_evaluator_total,
        "llm_evaluator_total": llm_evaluator_total,
    }


# 실행 예시
if __name__ == "__main__":
    # 인간 평가자 데이터
    human_data = {
        # khloe (evaluator1)
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
        # noah (evaluator2)
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

    # 두 평가자의 점수 평균
    data = calculate_human_average_and_store(human_data)

    # 결과 확인
    print("\n=== 인간 평가 평균 ===")
    print("data['human_evaluator']:")
    for item, scores in data["human_evaluator"].items():
        print(f"  {item}: {scores}")

    # LLM 평가 점수
    data["llm_evaluator"] = {
        "headline_attention": [
            4.0,
            2.0,
            2.0,
            4.0,
            4.0,
            4.0,
            5.0,
            5.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
            4.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ],
        "content_interest": [
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            4.0,
            5.0,
            4.0,
            4.0,
            3.0,
            4.0,
            4.0,
            2.0,
            3.0,
            3.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ],
        "content_readability": [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            2.0,
            3.0,
            2.0,
            2.0,
            3.0,
            2.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ],
        "concept_completeness": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            3.0,
            1.0,
            3.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ],
    }
    print("\n=== LLM 평가 결과 ===")
    print("data['llm_evaluator']:")
    for item, scores in data["llm_evaluator"].items():
        print(f"  {item}: {scores}")

    # 상관계수 계산
    correlations, total_correlation = calculate_pearson_correlation(data)
