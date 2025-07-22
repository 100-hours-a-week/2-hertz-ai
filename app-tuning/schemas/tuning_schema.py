"""
매칭 관련 데이터 모델 정의
사용자 간 매칭 요청 및 응답에 사용되는 Pydantic 모델
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TuningMatchingList(BaseModel):
    userIdList: List[int]


class TuningMatching(BaseModel):
    """
    튜닝(매칭) id 리스트 모델

    userId: int = Field(..., description="매칭할 사용자의 ID")
    model_config = Field(json_schema_extra={"example": {"userId": 1}})
    """

    userIdList: List[int] = Field(..., description="추천된 사용자 ID 목록")


class TuningResponse(BaseModel):
    code: str = Field(..., description="응답 코드 (매칭 성공 여부)")
    data: Optional[TuningMatchingList] = Field(
        None, description="매칭된 사용자 ID 목록"
    )

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
