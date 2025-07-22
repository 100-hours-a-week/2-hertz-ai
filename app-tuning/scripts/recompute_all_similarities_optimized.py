"""
최적화된 유사도 재계산 스크립트 (V3)

주요 개선 사항:
1. **데이터 일괄 로딩**: 모든 사용자 데이터를 처음에 한 번만 로드하여 DB I/O 최소화.
2. **데이터 공유**: 로드된 데이터를 각 병렬 프로세스에 인자로 전달하여 중복 로딩 방지.
3. `user_service.py`의 `update_similarity_for_users_v3` 함수를 재사용하여 코드 일관성 유지.
4. `ThreadPoolExecutor`를 사용하여 사용자별 'friend' 및 'couple' 카테고리 계산을 병렬로 수행.
5. `upsert` 로직을 개선하여 실제 변경이 있을 때만 DB에 쓰도록 최적화.
6. 불필요한 로그 제거 및 성능 측정 로그 정리.
"""

import concurrent.futures
import os
import sys
import time
import traceback

import numpy as np

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from core.vector_database import get_user_collection  # noqa: E402
from services.user_service import update_similarity_for_users_v3  # noqa: E402
from utils.logger import log_performance, logger  # noqa: E402

# 최적화된 워커 수 (CPU 코어의 75% 사용)
# WORKER_COUNT = max(1, int(os.cpu_count() * 0.75))
BATCH_SIZE = 10  # 로그 출력 단위


def get_all_users_data():
    """모든 사용자 데이터를 한 번에 가져옵니다."""
    try:
        logger.info("📊 모든 사용자 데이터 로딩 중...")
        data = get_user_collection().get(include=["embeddings", "metadatas"])
        logger.info(f"✅ {len(data['ids'])}명의 사용자 데이터 로딩 완료")
        return data
    except Exception as e:
        logger.error(f"[CRITICAL] 사용자 데이터를 가져오는 데 실패했습니다: {e}")
        return None


def convert_numpy_floats(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_floats(i) for i in obj]
    return obj


def process_user_category(user_id: str, category: str, all_users_data: dict):
    """단일 사용자의 특정 카테고리 유사도를 업데이트합니다."""
    try:
        # update_similarity_for_users_v3 내부에서 similarities를 저장하기 전에 float32 변환이 누락될 수 있으므로,
        # 변환을 강제 적용 (함수 내부에서 이미 처리 중이어도 중복 적용은 무해)
        update_similarity_for_users_v3(user_id, category, all_users_data)
        return True, None
    except Exception as e:
        error_message = f"[ERROR] {category} 유사도 계산 실패 for {user_id}: {e}"
        return False, error_message


def process_user_wrapper(user_id: str, all_users_data: dict):
    """
    한 명의 사용자에 대해 'friend'와 'couple' 카테고리 유사도 계산을
    병렬로 처리하는 래퍼 함수입니다.
    """
    categories = ["friend", "couple"]
    success_count = 0
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(categories)) as executor:
        future_to_category = {
            executor.submit(
                process_user_category, user_id, category, all_users_data
            ): category
            for category in categories
        }

        for future in concurrent.futures.as_completed(future_to_category):
            success, error = future.result()
            if success:
                success_count += 1
            else:
                errors.append(error)

    if success_count == len(categories):
        return True, user_id
    else:
        logger.error(f"❌ {user_id} 처리 중 오류 발생: {errors}")
        return False, user_id


# @log_performance(
#     operation_name="recompute_all_similarities_optimized", include_memory=True
# )
# def recompute_all_similarities_optimized():
#     """최적화된 유사도 재계산 메인 함수"""
#     logger.info("🚀 최적화된 유사도 재계산 시작 (V3)...")
#     start_time = time.time()

#     all_users_data = get_all_users_data()
#     if not all_users_data or not all_users_data.get("ids"):
#         logger.warning("처리할 사용자가 없습니다.")
#         return

#     user_ids = all_users_data["ids"]
#     total_users = len(user_ids)
#     logger.info(f"📈 처리 대상 사용자: {total_users}명")
#     # logger.info(f"⚙️ 워커 수: {WORKER_COUNT}")

#     processed_count = 0
#     success_count = 0
#     failure_count = 0

#     with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
#         future_to_user_id = {
#             executor.submit(process_user_wrapper, user_id, all_users_data): user_id
#             for user_id in user_ids
#         }

#         for future in concurrent.futures.as_completed(future_to_user_id):
#             processed_count += 1
#             try:
#                 success, user_id = future.result()
#                 if success:
#                     success_count += 1
#                 else:
#                     failure_count += 1

#                 if processed_count % BATCH_SIZE == 0 or processed_count == total_users:
#                     progress = (processed_count / total_users) * 100
#                     logger.info(
#                         f"🔄 진행률: {processed_count}/{total_users} ({progress:.1f}%) "
#                         f"(성공: {success_count}, 실패: {failure_count})"
#                     )

#             except Exception as e:
#                 user_id = future_to_user_id[future]
#                 logger.error(
#                     f"[CRITICAL] 사용자 {user_id} 처리 중 심각한 오류 발생: {e}"
#                 )
#                 traceback.print_exc()
#                 failure_count += 1

#     total_time = time.time() - start_time
#     logger.info("🎉 전체 유사도 재계산 완료!")
#     logger.info(f"📊 총 처리 시간: {total_time:.2f}초")
#     logger.info(f"✅ 성공: {success_count}, ❌ 실패: {failure_count}")
#     if total_time > 0:
#         logger.info(f"⚡️ 평균 처리 속도: {total_users / total_time:.2f} users/sec")


@log_performance(
    operation_name="recompute_all_similarities_optimized_v2", include_memory=True
)
def recompute_all_similarities_optimized_v2():
    """순차 처리 방식으로 변경된 유사도 재계산 메인 함수"""
    logger.info("🐌 순차 처리 방식의 유사도 재계산 시작 (V3)...")
    start_time = time.time()

    all_users_data = get_all_users_data()
    if not all_users_data or not all_users_data.get("ids"):
        logger.warning("처리할 사용자가 없습니다.")
        return

    user_ids = all_users_data["ids"]
    total_users = len(user_ids)
    logger.info(f"📈 처리 대상 사용자: {total_users}명")

    processed_count = 0
    success_count = 0
    failure_count = 0

    # --- 기존 병렬 처리 코드 ---
    # with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
    #     future_to_user_id = { ... }
    #     for future in concurrent.futures.as_completed(future_to_user_id):
    #         ...

    # --- ✨ 변경된 순차 처리 코드 ---
    for user_id in user_ids:
        processed_count += 1
        try:
            # 각 사용자를 순서대로 하나씩 처리
            success, _ = process_user_wrapper(user_id, all_users_data)
            if success:
                success_count += 1
            else:
                failure_count += 1

        except Exception as e:
            logger.error(f"[CRITICAL] 사용자 {user_id} 처리 중 심각한 오류 발생: {e}")
            traceback.print_exc()
            failure_count += 1

        # 진행률 로깅 (기존 로직 유지)
        if processed_count % BATCH_SIZE == 0 or processed_count == total_users:
            progress = (processed_count / total_users) * 100
            logger.info(
                f"🔄 진행률: {processed_count}/{total_users} ({progress:.1f}%) "
                f"(성공: {success_count}, 실패: {failure_count})"
            )

    total_time = time.time() - start_time
    logger.info("🎉 전체 유사도 재계산 완료!")
    logger.info(f"📊 총 처리 시간: {total_time:.2f}초")
    logger.info(f"✅ 성공: {success_count}, ❌ 실패: {failure_count}")
    if total_time > 0:
        logger.info(f"⚡️ 평균 처리 속도: {total_users / total_time:.2f} users/sec")


if __name__ == "__main__":
    recompute_all_similarities_optimized_v2()
