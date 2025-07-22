"""
사용자 등록 API 엔드포인트 정의 및 관리
사용자 개인정보, 관심사, 키워드를 받아 임베딩 벡터를 생성하고 벡터 DB에 저장
/api 요청을 처리하고, 비즈니스 로직 실행을 위해 컨트롤러와 연결
"""

from api.controllers import user_controller
from fastapi import APIRouter, Body, Path
from schemas.user_schema import BaseResponse, EmbeddingRegister


class UserRouter:
    """
    사용자 등록 및 사용자 관련 엔드포인트를 처리하는 라우터 클래스
    사용자 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(prefix="/api", tags=["개발자용 API"])
        self.router_v1 = APIRouter(prefix="/api/v1", tags=["v1"])
        self.router_v2 = APIRouter(prefix="/api/v2", tags=["v2"])
        self.router_v3 = APIRouter(prefix="/api/v3", tags=["v3"])
        # 엔드포인트 등록

        self.router.add_api_route(
            "/v1/users",
            self.db_user_list,
            methods=["GET"],
            response_model=BaseResponse,
            summary="사용자 조회(내부 확인용)",
            description="벡터DB user_collection에 등록된 사용자 리스트를 조회합니다.",
        )

        self.router.add_api_route(
            "/v1/similarities",
            self.db_similarity_list,
            methods=["GET"],
            response_model=BaseResponse,
            summary="매칭 스코어 조회(내부 확인용)",
            description="벡터DB similarity_collection에 등록된 모든 매칭 스코어를 조회합니다.",
        )
        self.router.add_api_route(
            "/v3/similarities",
            self.db_similarity_list_v3,
            methods=["GET"],
            response_model=BaseResponse,
            summary="매칭 스코어 조회(내부 확인용)",
            description="벡터DB 유사도 collection에 등록된 모든 매칭 스코어를 조회합니다.",
        )

        self.router.add_api_route(
            "/v1/reset/chromadb",
            self.db_reset_data,
            methods=["DELETE"],
            response_model=BaseResponse,
            summary="(테스트 환경 리셋을 전체 DB 초기화)",
            description="<주의>벡터DB-collection에 저장된 모든 데이터를 초기화 합니다.",
        )

        self.router_v1.add_api_route(
            "/users",
            self.create_user,
            methods=["POST"],
            response_model=BaseResponse,
            summary="사용자 등록",
            description="사용자 등록 후 임베딩 벡터를 생성합니다.",
        )

        self.router_v2.add_api_route(
            "/users/{user_id}",
            self.delete_user_data,
            methods=["DELETE"],
            response_model=BaseResponse,
            summary="사용자 삭제",
            description="벡터 데이터베이스에서 사용자 데이터를 삭제합니다.",
        )

        # -------------------------v3-------------------------------

        self.router_v3.add_api_route(
            "/users",
            self.create_user_v3,
            methods=["POST"],
            response_model=BaseResponse,
            summary="사용자 등록(v3)",
            description="사용자 등록 후 임베딩 벡터를 생성합니다.",
        )
        self.router_v3.add_api_route(
            "/users/{user_id}",
            self.delete_user_data_v3,
            methods=["DELETE"],
            response_model=BaseResponse,
            summary="사용자 삭제(v3)",
            description="벡터 데이터베이스에서 사용자 데이터를 삭제합니다.",
        )

    async def db_user_list(self) -> BaseResponse:
        return await user_controller.db_user_list()

    async def db_similarity_list(self) -> BaseResponse:
        return await user_controller.db_similarity_list()

    async def db_reset_data(self) -> BaseResponse:
        return await user_controller.db_reset_data()

    async def db_similarity_list_v3(self, category) -> BaseResponse:
        return await user_controller.db_similarity_list_v3(category)

    async def create_user_v3(
        self, user_data: EmbeddingRegister = Body(..., description="사용자 등록 데이터")
    ) -> BaseResponse:
        return await user_controller.create_user_v3(user_data)

    async def delete_user_data_v3(
        self, user_id: int = Path(..., description="삭제할 사용자의 ID")
    ) -> BaseResponse:
        return await user_controller.delete_user_data_v3(user_id)

    # -------------------- 아래는 기존 버전----------------------
    async def create_user(
        self, user_data: EmbeddingRegister = Body(..., description="사용자 등록 데이터")
    ) -> BaseResponse:
        """
        신규 사용자를 등록 후 임베딩 벡터 생성

        - **user_data**: 사용자 등록 정보 (개인정보, 키워드, 관심사 등)

        **응답 예시**:
        ```json
        {
          "code": "EMBEDDING_REGISTER_SUCCESS",
          "data": null
        }
        ```

        **오류 응답**:
        - 409 Conflict: 이미 존재하는 사용자 ID
        ```json
        {
          "code": "EMBEDDING_REGISTER_CONFLICT",
          "data": null
        }
        ```
        """
        return await user_controller.create_user(user_data)

    async def delete_user_data(
        self, user_id: int = Path(..., description="삭제할 사용자의 ID")
    ) -> BaseResponse:
        """
        사용자 데이터 삭제

        - **user_id**: 사용자 아이디

        **응답 예시**:
        ```json
        {
          "code": "EMBEDDING_DELETE_SUCCESS",
          "data": null
        }
        ```


        **오류 응답**:
        - 404 Conflict: 없는 사용자 데이터 삭제 시 오류
        ```json
        {
          "code": "EMBEDDING_DELETE_NOT_FOUND_USER",
          "data": null
        }
        ```
        """
        return await user_controller.delete_user_data(user_id)
