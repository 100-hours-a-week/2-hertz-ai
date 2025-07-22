import os
import re
import time
from datetime import datetime

import google.generativeai as genai
import pandas as pd
from candidates_ver2 import SAMPLE_NEWS
from dotenv import load_dotenv
from scipy.stats import pearsonr

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


def create_evaluation_prompt(sample_data):
    """평가용 프롬프트 생성"""
    sample_json = (
        f'{{"title": "{sample_data["title"]}", "content": "{sample_data["content"]}"}}'
    )

    prompt = f"""
당신은 소셜 디스커버리 앱의 콘텐츠 품질 평가 전문가입니다. 
생성된 매칭 뉴스가 사용자의 시선을 끌고, 재미와 흥미를 유발하는지 아래의 명확한 기준에 따라 평가합니다.

## 평가지표 (총 15점)

### 1. 안전성 체크 (Pass / Fail)
이 항목은 점수화되지 않으며, '부적합' 판정 시 평가를 즉시 중단하고 그 사유를 명시합니다.
생성된 콘텐츠에 욕설, 차별, 혐오, 폭력성, 선정성 등 사용자에게 불쾌감을 주거나 사회적으로 부적절한 내용이 포함되어 있는가?

**판정 기준:**
- [적합]: 안전 기준을 통과함. 아래 평가를 계속 진행합니다.
- [부적합]: 안전 기준 미달. (예: 혐오 표현 포함, 과도한 선정성 등)


### 2. 헤드라인 주목도 (5점)
첫인상과 호기심 유발 능력 평가

**점수 기준:**
- 5점: 제목만 봐도 바로 클릭하고 싶음. 강력한 호기심 유발
   - 예시: "🚨[세계 대혼란] ESTP × ENFP, 대화 424번? 지금 지구에 무슨 일이?!"
   - 특징: [단독], [속보] 태그 + 과장된 표현 + 강한 이모지 + 충격적 문구
- 4점: 제목이 흥미롭고 읽어보고 싶음
   - 예시: "📢 [속보] ESTP와 ENFP, 이 조합 실화냐?"
   - 특징: 적절한 임팩트 + 호기심 유발 표현 + 구체적 상황 암시
- 3점: 제목이 나쁘지 않아서 읽어볼 만함
   - 예시: "ESTP와 ENFP의 특별한 우정 이야기"
   - 특징: 평범하지만 부정적이지 않음 + 기본 정보 포함
- 2점: 제목이 평범해서 별로 관심 안 생김
   - 예시: "ESTP와 ENFP 성격 분석 결과"
   - 특징: 딱딱하고 일반적 + 재미 요소 부족 + 공식적 느낌
- 1점: 제목이 지루하거나 매력 없음
   - 예시: "두 사용자 간 커뮤니케이션 패턴 연구"
   - 특징: 학술적/기계적 표현 + 무미건조 + 접근성 제로


### 3. 본문 재미 (5점)
스토리의 흡입력, 표현의 신선함, 전반적인 즐거움 종합적 평가

**점수 기준:**
- 5점: 계속 웃으면서 읽었고, 시간 가는 줄 모름. 완전히 새로운 방식이라 놀람**
   - 예시: "ESTP: '야 나 생각났어. 지금 당장 하자.' ENFP: '헐 대박! 나도 방금 그 생각함!' → 그리고 둘이 만남 → 세계관 충돌 → 빅뱅 발생 💥"
   - 특징: 창의적 대화 구성 + 상상력 넘치는 표현 + 계속 웃음 유발 + 독창적 스토리텔링
- 4점: 여러 번 웃었고 집중해서 읽음. 독창적이고 재밌는 부분들이 많음**
   - 예시: "이 정도면 그냥 형제 아님? 대화량이 곧 텐션이라는 걸 보여주는 사례. 서로 피드백 주고받고, 투닥거리다가 웃고, 인생 얘기까지."
   - 특징: 재밌는 표현 다수 + 적절한 유머 + 몰입감 있는 구성 + 어느 정도 참신함
- 3점: 가끔 재밌고 나름 집중해서 읽음. 어느 정도 새로운 요소 있음**
   - 예시: "ISTJ의 완벽한 시간표와 ENFP의 번개같은 여행, 이 둘이 만나면 세상 어떤 프로젝트든 성공으로 이끌 수 있다는 걸 보여주는 완벽한 콜라보였다."
   - 특징: 기본적 재미 + 읽을 만한 수준 + 일반적이지만 나쁘지 않은 구성
- 2점: 약간 재밌긴 하지만 밋밋함. 중간에 지루한 구간 있음**
   - 예시: "두 사람은 MBTI가 다르지만 공통 관심사가 많아서 좋은 친구가 될 것 같습니다. 브런치와 독서를 좋아한다는 점이 인상적입니다."
   - 특징: 약간의 재미 + 예측 가능한 내용 + 밋밋한 전개 + 집중력 저하 구간
- 1점: 별로 재밌지 않고 지루함. 뻔하고 예측 가능한 내용**
   - 예시: "사용자 A는 ESTP 성향을 보이며, 사용자 B는 ENFP 특성을 나타냅니다. 두 성격 유형의 특징을 분석하면 다음과 같습니다."
   - 특징: 재미 요소 거의 없음 + 딱딱한 분석 위주 + 뻔한 구성 + 지루함


### 4. 본문 가독성 (3점)
시각적 구성과 읽기 편한 정도 평가

[가독성 분석 가이드]
아래 1, 2번 항목을 순서대로 분석하고, 그 결과를 종합하여 최종 점수를 부여하세요.
1. 글의 구조 (Structure):
   - 문단이 너무 길지 않게 잘 나뉘어 있는가? 통으로 된 '텍스트 덩어리'는 없는가?
   - 이 기준에 미달하면 다른 것을 볼 필요 없이 즉시 1점을 부여하세요.
2. 본문 스타일링 (Styling in Body Text):
   - 소제목뿐만 아니라, 본문 내용 자체에도 이모지나 볼드체 같은 스타일링 요소가 적절히 사용되어 글에 생동감을 주는가?

**점수 기준:**
- 3점 (매우 뛰어남): **[분석 가이드]**의 1번(구조)과 2번(본문 스타일링)을 모두 완벽하게 충족. 구조가 안정적이고, 본문에도 스타일링 요소가 잘 녹아있어 읽는 재미가 있음.
- 2점 (보통): **[분석 가이드]**의 1번(구조)은 충족하지만, 2번(본문 스타일링)이 부족함. 즉, 글의 전체적인 구조는 좋으나, 본문 내용 자체에는 이모지나 강조가 거의 사용되지 않아 다소 밋밋하고 심심하게 느껴짐.
- 1점 (개선 필요): **[분석 가이드]**의 1번(구조)부터 문제가 있음. 문단 구분이 엉망이거나, 과도한 효과 남용으로 읽기 불편함.


### 5. 컨셉 완성도 (3점)
글의 창의적인 '컨셉'을 설정하고, 그 컨셉을 얼마나 일관되고 완벽하게 실행했는가?

[컨셉 완성도 분석 가이드]
이 항목은 단순히 문법이나 어미 통일을 넘어, 글 전체가 하나의 창의적인 아이디어를 얼마나 효과적으로 구현했는지 평가합니다.

- **🎯 평가 핵심: 이 글이 가진 '한 방(컨셉)'은 무엇이며, 그 컨셉을 끝까지 밀고 나갔는가?**

**점수 기준:**

- **3점 (매우 뛰어남: 완벽한 컨셉 실행)**
   - **정의:** 창의적인 컨셉을 설정하고, **글의 모든 요소(제목, 소제목, 문장, 이모지, 비유 등)가 그 컨셉을 강화**하는 데 완벽하게 기여함. 기술적 완벽함을 넘어 감탄을 자아내는 수준.
   - **이런 느낌일 때 3점:** "이 글, '컨셉' 한 번 제대로 잡았는데? 모든 내용이 그 컨셉을 위해 미친 듯이 달려드네!"
   - **대표 예시:** '두 사람의 만남을 우주적 대사건으로 과장하는 컨셉'을 설정하고, '빅뱅 발생', '넷플릭스 입장' 등 모든 요소를 이 컨셉에 맞춰 일관되게 실행함.
- **2점 (보통: 안정적인 글쓰기)**
   - **정의:** 글의 톤앤매너는 일관되지만, 3점처럼 글 전체를 관통하는 강력하고 창의적인 단일 '컨셉'은 부족함. 흠잡을 데 없이 잘 쓴 글이지만, 특별한 컨셉 없이 무난하게 느껴짐.
   - **이런 느낌일 때 2점:** "음, 글 잘 썼네. 흠잡을 데 없이 깔끔하고 일관돼. 근데 뭐랄까, 막 엄청 새롭거나 감탄스럽진 않네."
   - **대표 예시:** '매칭된 두 사람을 소개해주는 매거진'처럼 정보를 유용하게 잘 정리했지만, 독자를 압도하는 창의적인 컨셉보다는 안정적인 정보 전달에 초점을 맞춤.
- **1점 (개선 필요: 컨셉 붕괴)**
   - **정의:** 설정한 컨셉을 유지하지 못하거나, **글의 핵심 주제와 관련 없는 정보(예: 날씨, 자격증 취득 방법 등)가 갑자기 끼어들어** 글의 흐름과 통일성을 심각하게 해침.
   - **이런 느낌일 때 1점:** "아니, 갑자기 여기서 이 얘기가 왜 나와? 글의 컨셉을 완전히 잃어버렸잖아."
   - **대표 예시:** '두 사람의 우정'이라는 컨셉으로 글을 진행하다가, 맥락과 무관한 '오늘의 서울 날씨' 정보를 상세히 나열하여 컨셉의 통일성을 스스로 무너뜨림.

---

## 📝 평가 대상 텍스트
{sample_json}

---

## 📋 출력 형식
반드시 다음 형식으로만 답변해주세요. 다른 설명 없이 정확히 이 형식을 따라주세요:

**평가 결과:**
### 1. 안전성 체크
- [적합/부적합] 
- [사유 기재]
(안전성 '적합' 시에만 아래 내용 출력)

### 2. 헤드라인 주목도: (점수)/5점
- 평가 근거: [왜 이 점수를 부여했는지 구체적인 이유 서술.]
- 개선 제안: [더 높은 점수를 위한 헤드라인 아이디어 1가지 제안]

### 3. 본문 재미: (점수)/5점
- 평가 근거: [글의 어떤 부분이 재미있었는지, 표현이 얼마나 참신했는지 등을 종합적으로 서술.]
- 개선 제안: [내용을 더 매력적으로 만들 구체적인 아이디어 제안.]

### 4. 본문 가독성: (점수)/3점
- 평가 근거: [시각적 구성(줄 바꿈, 이모지 등)의 장단점을 구체적으로 서술.]
- 개선 제안: [가독성을 높일 수 있는 구체적인 서식 수정 제안.]

### 5. 컨셉 완성도: (점수)/3점
- 평가 근거: [이 글의 핵심 컨셉을 무엇으로 파악했는지 먼저 한 문장으로 정의하고, 그 컨셉이 왜 성공적이거나 실패했는지 구체적인 근거(본문 내용 인용)를 들어 설명.]
- 개선 제안: [컨셉을 더 매력적으로 만들거나, 붕괴된 컨셉을 바로잡기 위한 구체적인 액션 아이템을 제안.]

총점: (점수)/16점

종합 의견:

- 전체적 강점: [3가지]
- 주요 개선 영역: [3가지]
"""
    return prompt


def extract_scores(response_text):
    """응답에서 각 항목별 점수 추출"""
    scores = {}

    # 각 항목별 점수 추출
    patterns = {
        "headline_attention": r"2\.\s*헤드라인 주목도.*?:\s*(\d+(?:\.\d+)?)/5",
        "content_interest": r"3\.\s*본문 재미.*?:\s*(\d+(?:\.\d+)?)/5",
        "content_readability": r"4\.\s*본문 가독성.*?:\s*(\d+(?:\.\d+)?)/3",
        "concept_completeness": r"5\.\s*컨셉 완성도.*?:\s*(\d+(?:\.\d+)?)/3",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            scores[key] = float(match.group(1))
        else:
            scores[key] = None

    # 총점 추출
    total_pattern = r"총점:\s*(\d+(?:\.\d+)?)/16"
    total_match = re.search(total_pattern, response_text, re.IGNORECASE)
    scores["total"] = float(total_match.group(1)) if total_match else None

    return scores


def evaluate_single_sample(sample_data, retry_count=3):
    """단일 샘플 평가"""
    prompt = create_evaluation_prompt(sample_data)

    for attempt in range(retry_count):
        try:
            response = model.generate_content(prompt)
            scores = extract_scores(response.text)

            # 모든 점수가 성공적으로 추출되었는지 확인
            if all(score is not None for score in scores.values()):
                return scores
            else:
                print(f"점수 추출 실패 (시도 {attempt + 1}/{retry_count})")
                print(f"추출된 점수: {scores}")

        except Exception as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{retry_count}): {e}")
            time.sleep(2)

    return None


def evaluate_all_samples_by_llm(evaluator_name, sample_list):
    """한 평가자가 모든 샘플을 평가"""
    print(f"\n=== {evaluator_name} 평가 시작 ===")

    results = {
        "headline_attention": [],  # 헤드라인 주목도
        "content_interest": [],  # 본문 재미
        "content_readability": [],  # 본문 가독성
        "concept_completeness": [],  # 컨셉 완성도
    }

    for i, sample in enumerate(sample_list, 1):
        print(f"샘플 {i}/20 평가 중... ({sample.get('id', f'sample_{i}')})")

        scores = evaluate_single_sample(sample)

        if scores:
            results["headline_attention"].append(scores["headline_attention"])
            results["content_interest"].append(scores["content_interest"])
            results["content_readability"].append(scores["content_readability"])
            results["concept_completeness"].append(scores["concept_completeness"])

            print(
                f"✅ 완료 - 헤드라인 주목도:{scores['headline_attention']}, 본문 재미:{scores['content_interest']}, "
                f"본문 가독성:{scores['content_readability']}, 컨셉 완성도:{scores['concept_completeness']}"
            )
        else:
            print(f"❌ 평가 실패")
            # 실패시 None 추가
            for key in results:
                results[key].append(None)

        # API 호출 간격 조절
        time.sleep(1)

    return results


def main():

    # SAMPLE_NEWS에서 20개 샘플 준비
    all_samples = []
    for model_name, samples in SAMPLE_NEWS.items():
        # 각 모델에서 일정 수만큼 가져오기
        all_samples.extend(samples)

    llm_eval_results = evaluate_all_samples_by_llm("gemini-2.5-flash", all_samples)
    print(llm_eval_results)


if __name__ == "__main__":
    main()
