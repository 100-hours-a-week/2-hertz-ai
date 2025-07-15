import concurrent.futures
import os
import sys
import traceback

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from core.vector_database import get_user_collection  # noqa: E402
from services.user_service import update_similarity_for_users_v3  # noqa: E402
from utils.logger import log_performance, logger  # noqa: E402

# CPU 코어 수의 50%만 사용 (예: 8코어면 4개)
worker_count = max(1, int(os.cpu_count() * 0.5))


def get_all_user_ids():
    # user_similarities 컬렉션에서 유저 ID만 가져옴
    data = get_user_collection(collection_name="user_similarities").get(include=[])
    return data["ids"]


def process_user(user_id, category):

    try:
        update_similarity_for_users_v3(user_id, category)
        logger.info(f"[{category}] Updated: {user_id}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] {category} similarity failed for {user_id}: {e}")
        traceback.print_exc()
        return False


@log_performance(operation_name="recompute_all_similarities_v3", include_memory=True)
def recompute_all_similarities_v3():
    logger.info("✅ Recomputing similarities (sentence-based, v3)...")
    try:
        all_users = get_user_collection().get(include=["embeddings", "metadatas"])
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch all users: {e}")
        return

    user_ids = all_users["ids"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for user_id in user_ids:
            for category in ["friend", "couple"]:
                futures.append(executor.submit(process_user, user_id, category))

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as e:
                logger.error(f"[ERROR] Future failed: {e}")
            if i % 10 == 0:
                logger.info(f"진행률: {i}/{len(futures)}")

    logger.info("🎉 All similarity recomputations completed.")


if __name__ == "__main__":
    recompute_all_similarities_v3()


async def monitor_script_execution(process):
    """스크립트 실행 상태 모니터링"""
    try:
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info("✅ 유사도 재계산 스크립트 실행 완료")
            if stdout:
                logger.info(f"스크립트 출력: {stdout.decode()}")
        else:
            logger.error(
                f"❌ 유사도 재계산 스크립트 실행 실패 (exit code: {process.returncode})"
            )
            if stderr:
                logger.error(f"에러 출력: {stderr.decode()}")
    except Exception as e:
        logger.error(f"스크립트 모니터링 중 오류: {e}")
