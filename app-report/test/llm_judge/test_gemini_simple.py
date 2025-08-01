import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# 간단한 응답 생성 테스트
# response = model.generate_content(
#     "오버워치 아나라는 영웅을 금지시키면 바나나. 방금 농담을 10점 만점으로 평가해줘."
# )

# 튜닝리포트 형식 응답 생성 테스트(qwen, claude, chatgpt 순)
sample_news = [
    {
        """
        {
        "title": "[속보] 친구의 로맨스? 아니면 단순히 친구? 튜닝에서 만나는 두 사람의 우정 이야기!",
        "content": "\n\n**📰 [뉴스 헤드라인] 도입부**\n\n오늘은 튜닝에서 두 친구의 우정 이야기를 들려드리려고 합니다. 이름을 거론하지 않겠습니다만, 서로를 향해 '친구'라는 단어를 사용하는 두 사람의 이야기를 들어볼까요?\n\n**🧠 MBTI 궁합 분석**\n\n이 두 사람은 ENFP와 ISTJ의 궁합을 가지고 있습니다. ENFP의 자유로운 사고방식은 ISTJ의 좁은 시야를 넓혀주며, 감정 표현을 돕습니다. ISTJ는 ENFP에게 안정감과 신뢰를 제공하고, 현실적인 조언을 합니다. 두 사람은 함께 성장하며, 서로의 다름을 이해하고 존중합니다.\n\n**🎯 공통 관심사 발견**\n\n두 사람 모두 그림그리기를 좋아합니다. 그래서 오늘은 그림 그리기에 대한 이야기를 나눠볼까요? 요가도 함께 즐길 수 있는 활동이네요. 요가는 특히 이 날씨에 좋을 것 같습니다. \n\n**📱 대화 분석**\n\n두 사람이 주고 받은 대화 횟수가 85번으로, 이는 상당히 많은 대화를 나눈 것을 의미합니다. 이는 서로에게 매우 중요하며, 친구로서의 관계를 더욱 강화시키는 단계를 나타냅니다.\n\n**🔮 마무리**\n\n오늘은 요가 클래스에 참여하거나, 그림 그리는 모임에 가보는 것은 어떨까요? 그런 활동들은 서로의 친구 사이를 더욱 강화시킬 것입니다. 날씨가 맑고 바람은 부는 4mph 정도로, 외부 활동하기에 좋습니다. 요가는 특히 이 날씨에 좋을 것 같습니다. 혹시, 오늘은 그림 그리는 모임에 가보는 건 어떨까요? \n\n*Stay Tuned!*"
        }
        """
    },
    {
        """
        {
        "title": "🔥 [단독] 오늘의 최강 브로맨스가 밝혀졌다! 두 INTP의 평행이론, 현실로?",
        "content": "2025년 6월 17일, 서울\n📊 MBTI 궁합 분석\nINTP와 INTP의 만남이라... 마치 같은 주파수를 공유하는 두 뇌섹남녀의 우정 아닐까요? 둘 다 깊은 사고력과 독립적인 성향을 지닌 성찰가 타입으로, 말없이 함께 있어도 편안한 ‘무말(無말) 케미’가 기대됩니다. 심리 전문가들은 \"같은 MBTI 조합 중 INTP-INTP는 특히 지적 대화를 즐기며, 공감대가 형성되면 깊이 있는 관계로 발전할 가능성이 높다\"고 분석합니다. 단, 감정 표현이 서툰 만큼 서로의 신호에 민감해질 필요도 있겠네요.\n\n🎵/📺 공통 관심사 트렌드\n두 사람 모두 ‘베이킹’, ‘수집’, ‘스쿠버다이빙’, ‘요리’, ‘사진’, ‘무념무상(멍 때리기)’ 같은 취미를 공유하고 있어요. 이 조합, 생각보다 꽤 감각적이죠? 요즘 넷플릭스에서는 감성 다큐 ‘카페 온 더 비치’와 수중 촬영이 인상적인 ‘딥 다이브: 바다 아래의 진실’이 화제랍니다. 음악 차트는 여전히 핫해요:\n1위: NewJeans - 'Bubble Pop!'\n2위: ILLIT - 'Magnetic Garden'\n3위: 태연 - '비 오는 날엔 너'\n조용한 공간에서 따로 또 같이 듣기 좋은 라인업이네요!\n\n🌸 오늘의 데이트 팁\n오늘 서울의 날씨는 대체로 맑고, 낮 최고기온은 30도까지 오를 예정이라고 해요. 습도는 높지 않지만 자외선 지수는 꽤 강한 편이니 주의! 이런 날에는 실내 카페에서 멍하니 앉아 책을 읽거나, 반자동 에스프레소 머신 체험이 가능한 홈카페 클래스에 참여하는 것도 추천드립니다.\n\n👀 주변 반응\n같은 INTP끼리 통하는 이 조합, 이미 주변에서 ‘브레인 케미’로 불리고 있다고 하네요. “둘 다 대화할 때 갑자기 철학 얘기로 빠지는 거, 너무 웃겨요 ㅋㅋ”라는 팀원의 후문도 있었고, “카페 스터디한다고 했는데 두 시간 동안 한 마디도 안 하고 나왔다더라구요” 같은 목격담도 나왔다고 합니다. 궁금하시죠? 말 없이도 통하는 진짜 친구, 이런 거 아닐까요?\n\n🔮 다음 예고\n이 브로맨스, 조용히 깊어지는 중! 다음 매칭 뉴스에선 두 INTP가 도전한 ‘튜닝 스쿠버 챌린지’ 현장을 취재할 예정이에요. 어떤 대화 없는 감동이 기다리고 있을지, 다음 주도 기대해주세요!"
        }
        """
    },
    {
        """
        {
        "title": "🔥 [오늘의 매칭] ENFJ ✨ ENTP, 웃음과 에너지가 터졌다! 베이킹부터 넷플릭스까지 취향 합!",
        "content": "📰 오늘의 매칭 뉴스 - ENFJ와 ENTP, 찰떡 텐션 케미 확인\n\n오늘 매칭된 두 사람은 에너지 넘치는 ENFJ와 장난기 가득한 ENTP! 서로를 반사판 삼아 더 밝게 빛나는 궁합이라고 하네요. 둘 다 대화와 활동을 좋아하는 유형이라 처음 만남에서도 어색함 없이 스무스하게 케미를 타는 스타일이라고 합니다.\n\n🧠 MBTI 궁합 분석\nENFJ는 사람을 중심으로 생각하는 따뜻한 리더형, ENTP는 아이디어 뱅크 같은 발산형 탐색가! 정리하자면, ‘공감력 + 유머 + 추진력’의 삼박자를 갖춘 찐친 같은 연애 조합이에요. 관계 전문가에 따르면, “서로가 서로에게 끊임없는 자극이 되며, 재미있고 건강한 관계를 유지할 수 있는 대표 궁합 중 하나”라고 하네요.\n\n🎵 공통 관심사 기반 콘텐츠 추천\n두 사람 모두 ‘베이킹’, ‘인터넷 예능’, ‘넷플릭스’, ‘인테리어’, ‘댄스’, ‘메이크업’까지 다양한 감각적 취미를 갖고 있죠.\n- 요즘 베이킹 클래스 중 인기 1위는 ‘비건 스콘 만들기 with 디톡스 티’.\n- 넷플릭스 TOP3: ① 셰프의 방랑일지 시즌2, ② 너의 목소리가 들려 시즌2, ③ 픽셀하트: 게임 속 인연\n- 음원 차트 TOP3: 1위 NewJeans 'Bubble Pop!', 2위 ILLIT 'Magnetic Garden', 3위 태연 '비 오는 날엔 너'\n달콤한 디저트와 알콩달콩한 드라마라면, 오늘 이 커플의 플레이리스트는 이미 완성!\n\n🍿 데이트/우정 시나리오 제안\n이 조합에게 추천하는 오늘의 데이트 코스는 '홈 베이킹 + 리빙룸 시네마'! 함께 버터를 녹이고 반죽을 구우며 티격태격한 후, 구운 쿠키를 먹으며 넷플릭스 시청까지 이어지면 완벽한 하루가 될 것 같네요.\n또는 강남 신논현 근처 '도심 속 제과 실습 카페'를 함께 가는 것도 추천 코스입니다.\n\n🔮 날씨 기반 활동 팁 + 다음 매칭 예고\n오늘 서울은 30도까지 오르는 초여름 날씨로, 체감온도는 약 32도.\n햇볕은 강하지만 미세먼지는 '좋음', 야외보다는 실내 활동이 딱입니다.\n가벼운 린넨 셔츠나 통기성 좋은 면티가 딱이겠네요.\n\n다음 매칭 예고는 ‘강아지를 좋아하는 사람끼리의 썸 가능성’?! 기대되시죠?\n‘튜닝’ 매칭 뉴스는 내일도 이어집니다. Stay tuned!"
        }
        """
    },
]

prompt = f"""

공정한 심사위원 역할을 하여 아래 AI 응답에 대한 품질을 평가해 주세요. 
평가 시 응답의 몰입감, 정보의 풍부함, 스토리텔링, 재미, 창의성 등을 고려해야 합니다. 
평가를 시작하기 전에 간단한 설명을 제공하세요. 
가능한 한 객관적이어야 합니다. 
설명을 제공한 후, 다음 형식을 엄격히 따라 1~10점 척도로 응답을 평가해 주세요.
: "[점수]", 예시: "점수: [5]".

[AI 응답]
{sample_news[2]}

"""

response = model.generate_content(prompt)

print(response.text)
