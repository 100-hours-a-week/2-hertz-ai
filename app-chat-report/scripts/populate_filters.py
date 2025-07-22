import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)  
from datetime import datetime
from typing import List

from db.mongodb import mongodb
from utils.logger import logger


async def add_filter_words(words: List[str]):
    """
    MongoDB에 필터 단어 및 문장을 추가하거나 업데이트합니다.
    word 필드를 기준으로 upsert (존재하면 업데이트, 없으면 삽입) 합니다.
    """
    now = datetime.utcnow()
    filter_collection = mongodb.get_collection(
        "message_filters"
    )  # 함수 내에서 컬렉션 가져오기

    try:
        inserted_count = 0
        updated_count = 0
        for word_item in words:
            # word_item이 단순 문자열일 수도 있고, 더 복잡한 객체일 수도 있습니다.
            # 여기서는 단순 문자열이라고 가정합니다.
            result = filter_collection.update_one(
                {"word": word_item},  # 단어가 이미 존재하는지 확인
                {
                    "$set": {
                        "is_active": True, # 활성화 상태를 True로 설정
                        "updated_at": now,
                    },  # 활성화 및 업데이트 시간 설정
                    "$setOnInsert": {
                        "created_at": now
                    },  # 새로 삽입할 때만 생성 시간 설정
                },
                upsert=True,  # 문서가 없으면 삽입
            )
            if result.upserted_id:
                inserted_count += 1
            elif result.modified_count > 0:
                updated_count += 1

        logger.info(
            f"Successfully processed filter words. Inserted: {inserted_count}, Updated: {updated_count}."
        )
    except Exception as e:
        logger.error(f"Error adding filter words: {e}", exc_info=True)


async def populate_filters():
    """
    미리 정의된 필터 목록으로 MongoDB 컬렉션을 채웁니다.
    """
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
        # 여기에 추가하고 싶은 다른 단어/문장들을 더 넣으세요.
    ]
    logger.info("Starting to populate message filters...")
    await add_filter_words(filter_list)
    logger.info("Finished populating message filters.")


# 이 스크립트를 독립적으로 실행할 때
if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv

    load_dotenv()  # .env 파일 로드

    # MongoDB 연결 초기화가 필요합니다.
    # 만약 'db/mongodb.py'가 이미 전역 'mongodb' 인스턴스를 생성한다면,
    # 단순히 import하는 것만으로 충분할 수 있습니다.
    # 만약 아니라면, 여기서 MongoDB 클래스를 인스턴스화해야 합니다.
    # 예: from db.mongodb import MongoDB; mongodb_instance = MongoDB()

    asyncio.run(populate_filters())
