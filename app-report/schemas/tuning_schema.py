"""
매칭 관련 데이터 모델 정의
사용자 간 매칭 요청 및 응답에 사용되는 Pydantic 모델
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TuningMatching(BaseModel):
    """
    튜닝(매칭) id 리스트 모델

    userId: int = Field(..., description="매칭할 사용자의 ID")
    model_config = Field(json_schema_extra={"example": {"userId": 1}})
    """

    userIdList: List[int] = Field(..., description="추천된 사용자 ID 목록")


class TuningResponse(BaseModel):
    code: str = Field(..., description="응답 코드 (매칭 성공 여부)")
    data: Optional[TuningMatching] = Field(None, description="매칭된 사용자 ID 목록")

    """
    튜닝(매칭) 응답 모델
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": [
                {
                    "code": "TUNING_SUCCESS",
                    "data": {"userIdList": [30, 1, 5, 6, 99, 56]},
                },
                {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None},
            ]
        }
    )


class UserProfile(BaseModel):
    """
    사용자 프로필 모델 (튜닝 리포트 생성용)
    """

    gender: str = Field(..., description="성별")
    MBTI: str = Field(..., description="MBTI 분류")
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

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "MALE",
                "MBTI": "ISTJ",
                "religion": "NON_RELIGIOUS",
                "smoking": "NO_SMOKING",
                "drinking": "SOMETIMES",
                "personality": ["KIND", "INTROVERTED"],
                "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR", "PASSIONATE"],
                "currentInterests": ["BAKING", "DRAWING", "PLANT_PARENTING"],
                "favoriteFoods": ["FRUIT", "WESTERN", "STREET_FOOD"],
                "likedSports": ["BOWLING", "BILLIARDS", "YOGA"],
                "pets": ["FISH", "HAMSTER", "RABBIT"],
                "selfDevelopment": ["READING", "STUDYING", "CAFE_STUDY"],
                "hobbies": ["GAMING", "MUSIC"],
            }
        }


class TuningReport(BaseModel):
    """
    튜닝 리포트 생성 요청 모델
    """

    category: str = Field(..., description="매칭 유형")
    chatCount: int = Field(..., description="채팅 횟수")
    userA: UserProfile = Field(..., description="첫 번째 사용자 프로필")
    userB: UserProfile = Field(..., description="두 번째 사용자 프로필")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "FRIEND",
                "chatCount": 85,
                "userA": {
                    "gender": "MALE",
                    "MBTI": "ISTJ",
                    "religion": "NON_RELIGIOUS",
                    "smoking": "NO_SMOKING",
                    "drinking": "SOMETIMES",
                    "personality": ["KIND", "INTROVERTED"],
                    "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR", "PASSIONATE"],
                    "currentInterests": ["BAKING", "DRAWING", "PLANT_PARENTING"],
                    "favoriteFoods": ["FRUIT", "WESTERN", "STREET_FOOD"],
                    "likedSports": ["BOWLING", "BILLIARDS", "YOGA"],
                    "pets": ["FISH", "HAMSTER", "RABBIT"],
                    "selfDevelopment": ["READING", "STUDYING", "CAFE_STUDY"],
                    "hobbies": ["GAMING", "MUSIC"],
                },
                "userB": {
                    "gender": "FEMALE",
                    "MBTI": "ENFP",
                    "religion": "CHRISTIANITY",
                    "smoking": "SOMETIMES",
                    "drinking": "SOMETIMES",
                    "personality": ["NICE", "CALM"],
                    "preferredPeople": ["CUTE", "PASSIONATE"],
                    "currentInterests": ["NETFLIX", "DRAWING"],
                    "favoriteFoods": ["TTEOKBOKKI", "WESTERN", "BAKERY"],
                    "likedSports": ["GOLF", "YOGA"],
                    "pets": ["WANT_TO_HAVE"],
                    "selfDevelopment": ["READING", "DIET"],
                    "hobbies": ["OUTDOOR", "MUSIC", "INSTRUMENT"],
                },
            }
        }


class TuningReportResponse(BaseModel):
    """
    튜닝 리포트 생성 응답 모델
    """

    code: str = Field(..., description="응답 코드")
    data: dict[str, str] = Field(..., description="생성된 튜닝 리포트")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "TUNING_REPORT_SUCCESS",
                "data": {
                    "title": "📢 [속보] 누가 누구랑? 이번 주 새롭게 연결된 인연 공개!",
                    "content": "이번 주, 새로운 연결이 성사되었습니다!\n\n하지만… 누군지 바로 알려드릴 순 없죠😉\n\n지금부터 공개되는 힌트 3가지, 눈 크게 뜨고 확인하세요!",
                },
            }
        }
