import os
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from utils.logger import logger

load_dotenv()  # .env 파일 자동 로드


env = os.getenv("ENVIRONMENT")
if env == "dev":
    DB_NAME = os.getenv("MONGODB_DB_NAME_DEV")
else:
    DB_NAME = os.getenv("MONGODB_DB_NAME")

MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
if not MONGO_URI:
    raise ValueError(
        "MONGODB_CONNECTION_STRING 환경 변수가 없습니다. .env 파일을 확인해주세요."
    )


class MongoDB:
    def __init__(self):
        try:
            # 앱 시작 시 client를 초기화
            self.client = AsyncIOMotorClient(
                MONGO_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.db = self.client.get_database(DB_NAME)
            self.collection = self.db.get_collection("chat_reports")
            logger.info("🟢 MongoDB에 성공적으로 연결되었고, 헬스체크를 통과했습니다.")

        except ConnectionFailure as e:
            # 4. 연결 실패 시 에러 로깅 및 프로세스 종료
            logger.error(f"🔴 MongoDB 연결 실패: 헬스체크에 실패했습니다. 에러: {e}")
            raise  # 에러를 다시 발생시켜 FastAPI 시작을 중단

    def get_collection(self, collection_name: str) -> Any:
        """지정된 이름의 MongoDB 컬렉션 객체를 반환"""
        if not self.client:
            raise HTTPException(
                status_code=500,
                detail="MongoDB 연결이 초기화되지 않았습니다. 서버를 다시 시작해주세요.",
            )
        # Directly get the collection from the connected database
        return self.db.get_collection(collection_name)

    def close(self):
        # 앱 종료 시 연결을 닫음
        self.client.close()
        logger.info("⚪ MongoDB 연결이 해제되었습니다.")


# 전역 변수로 MongoDB 인스턴스 생성
mongodb = MongoDB()


# =================================================================
# 데이터베이스 함수
# =================================================================
def save_report_to_db(report_data: Dict[str, Any]) -> None:
    """MongoDB에 신고 데이터 저장"""
    try:
        chat_report_collection = mongodb.get_collection("chat_reports")
        chat_report_collection.insert_one(report_data)
        logger.info(
            f"Chat report saved: {report_data['messageId']}, "
            f"Result: {report_data['result']}, "
            f"Label: {report_data['label']}, "
            f"Confidence: {report_data['confidence']}"
        )
    except Exception as e:
        logger.error(f"MongoDB save failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


from datetime import datetime

from db.mongodb import mongodb  # Assuming this is your MongoDB client

message_filter_collection = mongodb.get_collection("message_filters")


# --- The filter population logic ---
async def add_filter_words(words: List[str]):
    """MongoDB에 필터 단어 및 문장을 추가하거나 업데이트합니다."""
    now = datetime.utcnow()
    filter_collection = mongodb.get_collection("message_filters")

    try:
        inserted_count = 0
        updated_count = 0
        for word_item in words:
            result = await filter_collection.update_one(  # <--- THIS IS NOW AWAITABLE
                {"word": word_item},
                {
                    "$set": {"is_active": True, "updated_at": now},
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            if result.upserted_id:  # Check for newly inserted document's ID
                inserted_count += 1
            elif result.modified_count > 0:  # Check for modified documents
                updated_count += 1

        logger.info(
            f"Successfully processed filter words. Inserted: {inserted_count}, Updated: {updated_count}."
        )
    except Exception as e:
        logger.error(f"Error adding filter words: {e}", exc_info=True)


async def populate_filters():
    """미리 정의된 필터 목록으로 MongoDB 컬렉션을 채웁니다."""
    filter_list = [
        "병신",
        "씨발",
        "뒤질래",
        "디질래",
        "씹",
        "지랄",
        "염병",
        "느금마",
        "개새끼",
        "새끼",
        "개병신",
        "좆같은",
        "좆밥",
        "병신새끼",
        "병신같은",
        "ㅗㅗ",
        "엿머겅",
        "엿",
        "바보",
        "멍청이",
        "정신병자",
        "fuck",
        "hell",
        "asshole",
        "shit",
        "shut up",
        "개소리",
        "닥쳐",
        "미친놈",
        "꺼져",
        "지옥",
        "젠장",
    ]
    logger.info("Starting to populate message filters...")
    await add_filter_words(filter_list)
    logger.info("Finished populating message filters.")


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    # The global 'mongodb' instance is already created by importing this module.
    # So we can directly run the async populate_filters function.
    asyncio.run(populate_filters())
