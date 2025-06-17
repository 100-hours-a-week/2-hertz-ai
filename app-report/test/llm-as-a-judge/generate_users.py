import random
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingRegister(BaseModel):
    userId: int = Field(..., description="사용자 식별용 ID", ge=1)
    emailDomain: str = Field(..., description="유저간 조직 구분용 이메일 도메인")
    gender: str = Field(..., description="성별 (MALE/FEMALE)")
    ageGroup: str = Field(..., description="연령대")
    MBTI: str = Field(..., description="MBTI", min_length=4)
    religion: str = Field(..., description="종교")
    smoking: str = Field(..., description="흡연 정도")
    drinking: str = Field(..., description="음주 정도")

    personality: List[str] = Field(..., description="본인의 성향")
    preferredPeople: List[str] = Field(..., description="선호하는 상대 성향")
    currentInterests: List[str] = Field(..., description="요즘 관심사")
    favoriteFoods: List[str] = Field(..., description="좋아하는 음식")
    likedSports: List[str] = Field(..., description="좋아하는 운동")
    pets: List[str] = Field(..., description="반려동물")
    selfDevelopment: List[str] = Field(..., description="자기계발")
    hobbies: List[str] = Field(..., description="취미")

    @model_validator(mode="after")
    def check_required_fields(self) -> "EmbeddingRegister":
        required_str_fields = [
            "emailDomain",
            "gender",
            "ageGroup",
            "MBTI",
            "religion",
            "smoking",
            "drinking",
        ]
        required_list_fields = [
            "personality",
            "preferredPeople",
            "currentInterests",
            "favoriteFoods",
            "likedSports",
            "pets",
            "selfDevelopment",
            "hobbies",
        ]

        for field in required_str_fields:
            value = getattr(self, field)
            if not value or not str(value).strip():
                raise ValueError(f"'{field}' must be a non-empty string")

        for field in required_list_fields:
            value = getattr(self, field)
            if not value or not isinstance(value, list) or len(value) == 0:
                raise ValueError(f"'{field}' must be a non-empty list")

        return self


class RandomUserGenerator:
    """랜덤 유저 데이터를 생성하는 클래스"""

    def __init__(self):
        # 각 필드별 후보 리스트 - ENUM 매핑을 기반으로 채워넣음
        self.email_domains = [
            "kakaotech.com",
            "example.com",
            "test.org",
            "company.co.kr",
            "startup.io",
        ]

        self.genders = ["MALE", "FEMALE"]

        self.age_groups = ["AGE_20S", "AGE_30S", "AGE_40S", "AGE_50S", "AGE_60_PLUS"]

        self.mbti_types = [
            "INTJ",
            "INTP",
            "ENTJ",
            "ENTP",
            "INFJ",
            "INFP",
            "ENFJ",
            "ENFP",
            "ISTJ",
            "ISFJ",
            "ESTJ",
            "ESFJ",
            "ISTP",
            "ISFP",
            "ESTP",
            "ESFP",
            "UNKNOWN",
        ]

        self.religions = [
            "NON_RELIGIOUS",
            "CHRISTIANITY",
            "BUDDHISM",
            "CATHOLICISM",
            "WON_BUDDHISM",
            "OTHER_RELIGION",
        ]

        self.smoking_levels = [
            "NO_SMOKING",
            "SOMETIMES",
            "EVERYDAY",
            "E_CIGARETTE",
            "TRYING_TO_QUIT",
        ]

        self.drinking_levels = [
            "NEVER",
            "ONLY_IF_NEEDED",
            "SOMETIMES",
            "OFTEN",
            "TRYING_TO_QUIT",
        ]

        self.personalities = [
            "CUTE",
            "RELIABLE",
            "SMILES_OFTEN",
            "DOESNT_SWEAR",
            "NICE_VOICE",
            "TALKATIVE",
            "GOOD_LISTENER",
            "ACTIVE",
            "QUIET",
            "PASSIONATE",
            "CALM",
            "WITTY",
            "POLITE",
            "SERIOUS",
            "UNIQUE",
            "FREE_SPIRITED",
            "METICULOUS",
            "SENSITIVE",
            "COOL",
            "SINCERE",
            "LOYAL",
            "OPEN_MINDED",
            "AFFECTIONATE",
            "CONSERVATIVE",
            "CONSIDERATE",
            "NEAT",
            "POSITIVE",
            "FRUGAL",
            "CHARACTERFUL",
            "HONEST",
            "PLAYFUL",
            "DILIGENT",
            "FAMILY_ORIENTED",
            "COMPETENT",
            "SELF_MANAGING",
            "RESPONSIVE",
            "WORKAHOLIC",
            "SOCIABLE",
            "LONER",
            "COMPETITIVE",
            "EMPATHETIC",
        ]

        self.preferred_people = [
            "CUTE",
            "RELIABLE",
            "SMILES_OFTEN",
            "DOESNT_SWEAR",
            "NICE_VOICE",
            "TALKATIVE",
            "GOOD_LISTENER",
            "ACTIVE",
            "QUIET",
            "PASSIONATE",
            "CALM",
            "WITTY",
            "POLITE",
            "SERIOUS",
            "UNIQUE",
            "FREE_SPIRITED",
            "METICULOUS",
            "SENSITIVE",
            "COOL",
            "SINCERE",
            "LOYAL",
            "OPEN_MINDED",
            "AFFECTIONATE",
            "CONSERVATIVE",
            "CONSIDERATE",
            "NEAT",
            "POSITIVE",
            "FRUGAL",
            "CHARACTERFUL",
            "HONEST",
            "PLAYFUL",
            "DILIGENT",
            "FAMILY_ORIENTED",
            "COMPETENT",
            "SELF_MANAGING",
            "RESPONSIVE",
            "WORKAHOLIC",
            "SOCIABLE",
            "LONER",
            "COMPETITIVE",
            "EMPATHETIC",
        ]

        self.current_interests = [
            "MOVIES",
            "NETFLIX",
            "VARIETY_SHOWS",
            "HOME_CAFE",
            "CHATTING",
            "DANCE",
            "SPACE_OUT",
            "COOKING",
            "BAKING",
            "DRAWING",
            "PLANT_PARENTING",
            "INSTRUMENT",
            "PHOTOGRAPHY",
            "FORTUNE_TELLING",
            "MAKEUP",
            "NAIL_ART",
            "INTERIOR",
            "CLEANING",
            "SCUBA_DIVING",
            "SKATEBOARDING",
            "SNEAKER_COLLECTION",
            "STOCKS",
            "CRYPTO",
        ]

        self.favorite_foods = [
            "TTEOKBOKKI",
            "MEXICAN",
            "CHINESE",
            "JAPANESE",
            "KOREAN",
            "VEGETARIAN",
            "MEAT_LOVER",
            "FRUIT",
            "WESTERN",
            "STREET_FOOD",
            "BAKERY",
            "HAMBURGER",
            "PIZZA",
            "BRUNCH",
            "ROOT_VEGETABLES",
            "CHICKEN",
            "VIETNAMESE",
            "SEAFOOD",
            "THAI",
            "SPICY_FOOD",
        ]

        self.liked_sports = [
            "BASEBALL",
            "SOCCER",
            "HIKING",
            "RUNNING",
            "GOLF",
            "GYM",
            "PILATES",
            "HOME_TRAINING",
            "CLIMBING",
            "CYCLING",
            "BOWLING",
            "BILLIARDS",
            "YOGA",
            "TENNIS",
            "SQUASH",
            "BADMINTON",
            "BASKETBALL",
            "SURFING",
            "CROSSFIT",
            "VOLLEYBALL",
            "PINGPONG",
            "FUTSAL",
            "FISHING",
            "SKI",
            "BOXING",
            "SNOWBOARD",
            "SHOOTING",
            "JIUJITSU",
            "SWIMMING",
            "MARATHON",
        ]

        self.pets = [
            "DOG",
            "CAT",
            "REPTILE",
            "AMPHIBIAN",
            "BIRD",
            "FISH",
            "LIKE_BUT_NOT_HAVE",
            "HAMSTER",
            "RABBIT",
            "NONE",
            "WANT_TO_HAVE",
        ]

        self.self_development = [
            "READING",
            "STUDYING",
            "CAFE_STUDY",
            "LICENSE_STUDY",
            "LANGUAGE_LEARNING",
            "INVESTING",
            "MIRACLE_MORNING",
            "CAREER_DEVELOPMENT",
            "DIET",
            "MINDFULNESS",
            "LIFE_OPTIMIZATION",
            "WRITING",
        ]

        self.hobbies = [
            "GAMING",
            "MUSIC",
            "OUTDOOR",
            "MOVIES",
            "DRAMA",
            "CHATTING",
            "SPACE_OUT",
            "APPRECIATION",
            "DANCE",
            "COOKING",
            "BAKING",
            "DRAWING",
            "PLANT_CARE",
            "INSTRUMENT",
            "PHOTOGRAPHY",
            "WEBTOON",
            "MAKEUP",
            "INTERIOR",
            "CLEANING",
            "SCUBA_DIVING",
            "COLLECTING",
            "STOCKS",
        ]

    def generate_random_user(self, user_id: int = None) -> EmbeddingRegister:
        """랜덤 유저 데이터를 생성합니다"""

        if user_id is None:
            user_id = random.randint(1, 10000)

        # 리스트 필드들은 1-4개 랜덤 선택
        personality = random.sample(self.personalities, random.randint(1, 10))
        preferred_people = random.sample(self.preferred_people, random.randint(1, 10))
        current_interests = random.sample(self.current_interests, random.randint(1, 10))
        favorite_foods = random.sample(self.favorite_foods, random.randint(1, 10))
        liked_sports = random.sample(self.liked_sports, random.randint(1, 10))
        pets = random.sample(self.pets, random.randint(1, 3))
        self_development = random.sample(self.self_development, random.randint(1, 10))
        hobbies = random.sample(self.hobbies, random.randint(1, 10))

        return EmbeddingRegister(
            userId=user_id,
            emailDomain=random.choice(self.email_domains),
            gender=random.choice(self.genders),
            ageGroup=random.choice(self.age_groups),
            MBTI=random.choice(self.mbti_types),
            religion=random.choice(self.religions),
            smoking=random.choice(self.smoking_levels),
            drinking=random.choice(self.drinking_levels),
            personality=personality,
            preferredPeople=preferred_people,
            currentInterests=current_interests,
            favoriteFoods=favorite_foods,
            likedSports=liked_sports,
            pets=pets,
            selfDevelopment=self_development,
            hobbies=hobbies,
        )

    def user_to_dict(self, user: EmbeddingRegister) -> dict:
        """유저 정보를 딕셔너리로 변환 (매칭 형식에 맞게)"""
        return {
            "gender": user.gender,
            "MBTI": user.MBTI,
            "religion": user.religion,
            "smoking": user.smoking,
            "drinking": user.drinking,
            "personality": user.personality,
            "preferredPeople": user.preferredPeople,
            "currentInterests": user.currentInterests,
            "favoriteFoods": user.favoriteFoods,
            "likedSports": user.likedSports,
            "pets": user.pets,
            "selfDevelopment": user.selfDevelopment,
            "hobbies": user.hobbies,
        }

    def extract_common_interests(
        self, userA: EmbeddingRegister, userB: EmbeddingRegister
    ) -> Dict[str, List[str]]:
        """
        두 사용자의 공통 관심사를 추출합니다.

        Args:
            userA: 첫 번째 사용자
            userB: 두 번째 사용자

        Returns:
            카테고리별 공통 관심사 딕셔너리
        """
        common_interests = {}

        # 리스트 필드들을 순회하면서 공통 항목 찾기
        list_fields = [
            ("personality", "성향"),
            ("preferredPeople", "선호하는 사람 유형"),
            ("currentInterests", "관심사"),
            ("favoriteFoods", "좋아하는 음식"),
            ("likedSports", "좋아하는 운동"),
            ("pets", "반려동물"),
            ("selfDevelopment", "자기계발"),
            ("hobbies", "취미"),
        ]

        for field_name, korean_name in list_fields:
            userA_items = set(getattr(userA, field_name))
            userB_items = set(getattr(userB, field_name))

            # 교집합 계산
            common = list(userA_items & userB_items)

            if common:  # 공통 항목이 있는 경우만 저장
                common_interests[korean_name] = common

        return common_interests

    def extract_common_interests_from_dict(
        self, userA_dict: dict, userB_dict: dict
    ) -> Dict[str, List[str]]:
        """
        딕셔너리 형태의 사용자 데이터에서 공통 관심사를 추출합니다.

        Args:
            userA_dict: 첫 번째 사용자 딕셔너리
            userB_dict: 두 번째 사용자 딕셔너리

        Returns:
            카테고리별 공통 관심사 딕셔너리
        """
        common_interests = {}

        # 리스트 필드들을 순회하면서 공통 항목 찾기
        list_fields = [
            ("personality", "성향"),
            ("preferredPeople", "선호하는 사람 유형"),
            ("currentInterests", "관심사"),
            ("favoriteFoods", "좋아하는 음식"),
            ("likedSports", "좋아하는 운동"),
            ("pets", "반려동물"),
            ("selfDevelopment", "자기계발"),
            ("hobbies", "취미"),
        ]

        for field_name, korean_name in list_fields:
            userA_items = set(userA_dict.get(field_name, []))
            userB_items = set(userB_dict.get(field_name, []))

            # 교집합 계산
            common = list(userA_items & userB_items)

            if common:  # 공통 항목이 있는 경우만 저장
                common_interests[korean_name] = common

        return common_interests

    def generate_matching_pair(
        self, category: str = "FRIEND", chat_counts: int = 100
    ) -> dict:
        """두 사용자의 매칭 페어를 생성하고 지정된 형식으로 반환"""
        userA = self.generate_random_user(1)
        userB = self.generate_random_user(2)

        common_interests = generator.extract_common_interests(userA, userB)
        print(f"\n공통 관심사: {common_interests}")

        return {
            "category": category,
            "chatCount": chat_counts,
            "userA": self.user_to_dict(userA),
            "userB": self.user_to_dict(userB),
        }

    def print_matching_pair_json(self):
        """매칭 페어를 JSON 형식으로 출력"""
        import json

        category = random.choice(["FRIEND", "COUPLE"])
        chat_counts = random.randint(10, 500)

        matching_data = self.generate_matching_pair(category, chat_counts)
        print(json.dumps(matching_data, ensure_ascii=False, indent=3))


# 사용 예시
if __name__ == "__main__":
    # 랜덤 유저 생성기 초기화
    generator = RandomUserGenerator()

    # 매칭 페어 JSON 형식으로 출력
    print("=== 매칭 페어 JSON 형식 출력 ===")
    generator.print_matching_pair_json()

    print("\n" + "=" * 60 + "\n")
