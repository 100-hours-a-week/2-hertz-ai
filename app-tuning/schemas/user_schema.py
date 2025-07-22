# 사용자 관련 데이터 모델(신규 등록 요청)

from typing import Dict, List, Optional

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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "userId": 1,
                "emailDomain": "kakaotech.com",
                "gender": "MALE",
                "ageGroup": "AGE_20S",
                "MBTI": "ESTP",
                "religion": "NON_RELIGIOUS",
                "smoking": "NO_SMOKING",
                "drinking": "SOMETIMES",
                "personality": ["NICE_VOICE", "CALM"],
                "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR", "PASSIONATE"],
                "currentInterests": ["BAKING", "DRAWING", "PLANT_PARENTING"],
                "favoriteFoods": ["FRUIT", "WESTERN", "STREET_FOOD"],
                "likedSports": ["BOWLING", "BILLIARDS", "YOGA"],
                "pets": ["FISH", "HAMSTER", "RABBIT"],
                "selfDevelopment": ["READING", "STUDYING", "CAFE_STUDY"],
                "hobbies": ["GAMING", "MUSIC"],
            }
        }
    )


class BaseResponse(BaseModel):
    """
    API 응답의 기본 구조
    """

    code: str = Field(..., description="응답 코드")
    data: Optional[Dict] = Field(None, description="응답 데이터 (있는 경우)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"code": "EMBEDDING_REGISTER_CREATED", "data": None}
        }
    )


class ErrorResponse(BaseModel):
    """
    API 오류 응답 구조
    """

    code: str = Field(..., description="오류 코드")
    data: Optional[Dict] = Field(None, description="오류 상세 정보 (있는 경우)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "EMBEDDING_REGISTER_BAD_REQUEST",
                "data": None,
            }
        }
    )


# v3 스키마
# class EmbeddingModify(BaseModel):
#     """
#     사용자 정보 수정 및 임베딩 재생성을 위한 요청 모델
#     """

#     userId: int = Field(..., description="사용자 식별용 ID")
#     emailDomain: str = Field(..., description="유저간 조직 구분용 이메일 도메인")
#     gender: str = Field(..., description="성별")
#     ageGroup: str = Field(..., description="연령대")
#     MBTI: str = Field(..., description="MBTI 분류")
#     religion: str = Field(..., description="종교")
#     smoking: str = Field(..., description="흡연 정도")
#     drinking: str = Field(..., description="음주 정도")
#     personality: List[str] = Field(..., description="본인의 성향")
#     preferredPeople: List[str] = Field(..., description="선호하는 상대 성향")
#     currentInterests: List[str] = Field(..., description="요즘 관심사")
#     favoriteFoods: List[str] = Field(..., description="좋아하는 음식")
#     likedSports: List[str] = Field(..., description="좋아하는 운동")
#     pets: List[str] = Field(..., description="반려동물")
#     selfDevelopment: List[str] = Field(..., description="자기계발")
#     hobbies: List[str] = Field(..., description="취미")

#     class Config:
#         schema_extra = {
#             "example": {
#                 "userId": 1,
#                 "emailDomain": "kakaotech.com",
#                 "gender": "MALE",
#                 "ageGroup": "AGE_20S",
#                 "MBTI": "ESTP",
#                 "religion": "NON_RELIGIOUS",
#                 "smoking": "NO_SMOKING",
#                 "drinking": "SOMETIMES",
#                 "personality": ["KIND", "INTROVERTED"],
#                 "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR", "PASSIONATE"],
#                 "currentInterests": ["BAKING", "DRAWING", "PLANT_PARENTING"],
#                 "favoriteFoods": ["FRUIT", "WESTERN", "STREET_FOOD"],
#                 "likedSports": ["BOWLING", "BILLIARDS", "YOGA"],
#                 "pets": ["FISH", "HAMSTER", "RABBIT"],
#                 "selfDevelopment": ["READING", "STUDYING", "CAFE_STUDY"],
#                 "hobbies": ["GAMING", "MUSIC"],
#             }
#         }


# class EmbeddingDelete(BaseModel):
#     """
#     사용자 및 임베딩 삭제를 위한 요청 모델
#     """

#     userId: int = Field(..., description="삭제할 사용자의 ID")

#     class Config:
#         schema_extra = {"example": {"userId": 1}}
