"""
사용자 매칭 추천 API 엔드포인트 정의 및 관리
사용자 ID를 받아 임베딩 벡터 코사인 유사도 기반으로 매칭 대상자 추천 리스트 반환
/api 요청을 처리하고, 비즈니스 로직 실행을 위해 컨트롤러와 연결
"""

from api.controllers import tuning_controller
from fastapi import APIRouter, Query
from schemas.tuning_schema import TuningResponse


class TuningRouter:
    """
    사용자 매칭 및 추천 관련 엔드포인트를 처리하는 라우터 클래스
    매칭 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])
        self.router_v3 = APIRouter(prefix="/api/v3", tags=["v3"])
        # 엔드포인트 등록
        self.router_v1.add_api_route(
            "/tuning",
            self.get_tuning,
            methods=["GET"],
            response_model=TuningResponse,
            summary="튜닝(추천) 리스트 조회",
            description="해당 사용자 정보를 기반으로 가장 매칭 확률이 높은 유저 리스트를 조회합니다.",
        )
        self.router_v3.add_api_route(
            "/tuning",
            self.get_tuning_by_category,
            methods=["GET"],
            response_model=TuningResponse,
            summary="카테고리별 튜닝(추천) 리스트 조회",
            description="해당 사용자 정보를 기반으로 가장 매칭 확률이 높은 유저 리스트를 조회합니다.",
        )

    async def get_tuning(
        self,
        user_id: int = Query(
            ..., alias="userId", description="매칭할 사용자의 ID", gt=0
        ),
    ) -> TuningResponse:
        return await tuning_controller.get_tuning_matches(user_id)

    async def get_tuning_by_category(
        self,
        user_id: int = Query(
            ..., alias="userId", description="매칭할 사용자의 ID", gt=0
        ),
        category: str = Query(..., description="카테고리 (v3용 파라미터)"),
    ) -> TuningResponse:
        return await tuning_controller.get_tuning_matches_by_category(user_id, category)
