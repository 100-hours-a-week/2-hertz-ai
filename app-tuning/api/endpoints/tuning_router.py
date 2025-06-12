"""
사용자 매칭 추천 API 엔드포인트 정의 및 관리
사용자 ID를 받아 임베딩 벡터 코사인 유사도 기반으로 매칭 대상자 추천 리스트 반환
/api 요청을 처리하고, 비즈니스 로직 실행을 위해 컨트롤러와 연결
"""

from fastapi import APIRouter, Query
from schemas.tuning_schema import TuningResponse

from ..controllers import tuning_controller


class TuningRouter:
    """
    사용자 매칭 및 추천 관련 엔드포인트를 처리하는 라우터 클래스
    매칭 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(prefix="/api", tags=["tuning"])
        # 엔드포인트 등록 (/api/v1/tuning)
        self.router.add_api_route(
            "/v1/tuning",
            self.get_tuning,
            methods=["GET"],
            response_model=TuningResponse,
            summary="튜닝(추천) 리스트 조회",
            description="해당 사용자 정보를 기반으로 가장 매칭 확률이 높은 유저 리스트를 조회합니다.",
        )

    async def get_tuning(
        self,
        user_id: int = Query(
            ..., alias="userId", description="매칭할 사용자의 ID", gt=0
        ),
    ) -> TuningResponse:
        """
        사용자 ID 기반 매칭 추천 제공

        - **user_id**: 매칭을 요청한 사용자의 ID (gt: 1 이상의 정수)

        **응답 예시**:
        ```json
        {
          "code": "TUNING_SUCCESS",
          "data": {
            "userIdList": [30, 1, 5, 6, 99, 56]
          }
        }
        ```

        매칭 결과가 없는 경우:
        ```json
        {
          "code": "TUNING_SUCCESS_BUT_NO_MATCH",
          "data": null
        }
        ```
        """
        return await tuning_controller.get_tuning_matches(user_id)
