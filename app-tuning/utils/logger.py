# 로깅 유틸리티(애플리케이션 로그 관리)

import asyncio
import functools
import logging
import os
import time
from datetime import datetime
from inspect import signature
from typing import Any, Callable, Dict, Optional

import psutil

# 로거 설정
logger = logging.getLogger("tuning_performance")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 파일 핸들러 (logs 디렉토리에 성능 로그 저장)
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(
    f"logs/performance_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)
logger.propagate = False  # 부모 로거로 메시지 전파 중단

# 성능 지표 컬렉션 (간단한 인메모리 저장소)
performance_metrics = {
    "api_response_times": {},
    "embedding_generation_times": [],
    "similarity_calculation_times": [],
    "db_operation_times": {},
    "error_counts": {},
    "memory_usage_samples": [],
    "memory_usage_by_function": {},
}


def log_performance(operation_name: Optional[str] = None, include_memory: bool = False):
    """
    성능 측정 및 로깅 데코레이터

    Args:
        operation_name: 로깅할 작업 이름 (기본값: 함수명)
        include_memory: 메모리 사용량도 함께 로깅할지 여부
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # 작업 이름 결정
            op_name = operation_name or func.__name__

            # 유저 ID 추출 시도
            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            user_id = (
                kwargs.get("user_id")
                or kwargs.get("userId")
                or (args[0] if args and isinstance(args[0], (str, int)) else None)
            )

            # 객체나 딕셔너리 내부 속성에서 추출
            if not user_id:
                for arg in bound_args.arguments.values():
                    if isinstance(arg, dict):
                        user_id = arg.get("user_id") or arg.get("userId")
                    elif hasattr(arg, "user_id"):
                        user_id = getattr(arg, "user_id")
                    elif hasattr(arg, "userId"):
                        user_id = getattr(arg, "userId")
                    if user_id:
                        break

            user_id = str(user_id) if user_id is not None else "unknown"
            # 카테고리 추출 시도 (파라미터로 지정된 경우 우선 사용)
            extracted_category = None
            if not extracted_category:
                # 1. kwargs에서 직접 추출
                extracted_category = kwargs.get("category")

                # 2. bound_args에서 category 파라미터 추출
                if not extracted_category and "category" in bound_args.arguments:
                    extracted_category = bound_args.arguments["category"]

                # 3. 객체나 딕셔너리 내부 속성에서 추출
                if not extracted_category:
                    for arg in bound_args.arguments.values():
                        if isinstance(arg, dict):
                            extracted_category = arg.get("category")
                        elif hasattr(arg, "category"):
                            extracted_category = getattr(arg, "category")
                        if extracted_category:
                            break
            category_info = (
                f", category={extracted_category}" if extracted_category else ""
            )

            # 초기 메모리 사용량
            if include_memory:
                initial_memory = _get_memory_usage()

            # 시작 시간 기록
            start_time = time.time()

            try:
                # 함수 실행
                result = await func(*args, **kwargs)

                # 경과 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 결과 정보 추출
                result_info = _extract_result_info(result)

                # 메모리 사용량 변화 계산
                memory_info = ""
                if include_memory:
                    final_memory = _get_memory_usage()
                    memory_diff = final_memory - initial_memory
                    memory_info = f", memory_diff={memory_diff:.2f}MB"

                    if op_name not in performance_metrics["memory_usage_by_function"]:
                        performance_metrics["memory_usage_by_function"][op_name] = []
                    performance_metrics["memory_usage_by_function"][op_name].append(
                        final_memory
                    )

                # 성능 정보 로깅
                logger.info(
                    f"PERF: {op_name} completed in {elapsed}s [userId={user_id}{category_info}{result_info}{memory_info}]"
                )

                # 메트릭 저장
                _store_metric(op_name, elapsed, result)

                return result
            except Exception as e:
                # 오류 발생 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 오류 정보 로깅
                error_type = type(e).__name__
                logger.error(
                    f"PERF-ERROR: {op_name} failed after {elapsed}s [userId={user_id}{category_info}, error_type={error_type}]"
                )

                # 오류 카운트 증가
                if op_name not in performance_metrics["error_counts"]:
                    performance_metrics["error_counts"][op_name] = {}

                if error_type not in performance_metrics["error_counts"][op_name]:
                    performance_metrics["error_counts"][op_name][error_type] = 0

                performance_metrics["error_counts"][op_name][error_type] += 1

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # 작업 이름 결정
            op_name = operation_name or func.__name__

            bound_args = signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            # 유저 ID 추출 시도
            user_id = (
                kwargs.get("user_id")
                or kwargs.get("userId")
                or (args[0] if args and isinstance(args[0], (str, int)) else None)
            )

            # 객체 속성 탐색
            if not user_id:
                for arg in bound_args.arguments.values():
                    if isinstance(arg, dict) and ("user_id" in arg or "userId" in arg):
                        user_id = arg.get("user_id") or arg.get("userId")
                        break
                    if hasattr(arg, "user_id"):
                        user_id = getattr(arg, "user_id")
                        break
                    if hasattr(arg, "userId"):
                        user_id = getattr(arg, "userId")
                        break

            user_id = str(user_id) if user_id is not None else "unknown"
            # 카테고리 추출 시도 (파라미터로 지정된 경우 우선 사용)
            extracted_category = None
            if not extracted_category:
                # 1. kwargs에서 직접 추출
                extracted_category = kwargs.get("category")

                # 2. bound_args에서 category 파라미터 추출
                if not extracted_category and "category" in bound_args.arguments:
                    extracted_category = bound_args.arguments["category"]

                # 3. 객체나 딕셔너리 내부 속성에서 추출
                if not extracted_category:
                    for arg in bound_args.arguments.values():
                        if isinstance(arg, dict):
                            extracted_category = arg.get("category")
                        elif hasattr(arg, "category"):
                            extracted_category = getattr(arg, "category")
                        if extracted_category:
                            break
            category_info = (
                f", category={extracted_category}" if extracted_category else ""
            )
            # 초기 메모리 사용량
            if include_memory:
                initial_memory = _get_memory_usage()

            # 시작 시간 기록
            start_time = time.time()

            try:
                # 함수 실행
                result = func(*args, **kwargs)

                # 경과 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 결과 정보 추출
                result_info = _extract_result_info(result)

                # 메모리 사용량 변화 계산
                memory_info = ""
                if include_memory:
                    final_memory = _get_memory_usage()
                    memory_diff = final_memory - initial_memory
                    memory_info = f", memory_diff={memory_diff:.2f}MB"
                    performance_metrics["memory_usage_samples"].append(final_memory)

                    if op_name not in performance_metrics["memory_usage_by_function"]:
                        performance_metrics["memory_usage_by_function"][op_name] = []
                    performance_metrics["memory_usage_by_function"][op_name].append(
                        final_memory
                    )
                # 성능 정보 로깅
                logger.info(
                    f"PERF: {op_name} completed in {elapsed}s [userId={user_id}{category_info}{result_info}{memory_info}]"
                )

                # 메트릭 저장
                _store_metric(op_name, elapsed, result)

                return result
            except Exception as e:
                # 오류 발생 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 오류 정보 로깅
                error_type = type(e).__name__
                logger.error(
                    f"PERF-ERROR: {op_name} failed after {elapsed}s [userId={user_id}{category_info}, error_type={error_type}]"
                )

                # 오류 카운트 증가
                if error_type not in performance_metrics["error_counts"]:
                    performance_metrics["error_counts"][error_type] = 0
                performance_metrics["error_counts"][error_type] += 1

                raise

        # 동기/비동기 함수에 맞는 래퍼 반환
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_db_operation(operation_type: str, collection_name: str) -> Callable:
    """
    데이터베이스 작업 성능 측정 데코레이터

    Args:
        operation_type: 작업 유형 (예: "get", "add", "upsert")
        collection_name: 컬렉션 이름
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_key = f"{collection_name}_{operation_type}"

            # 시작 시간 기록
            start_time = time.time()

            try:
                # 함수 실행
                result = func(*args, **kwargs)

                # 경과 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 작업 데이터 크기 추정
                data_info = ""
                if "ids" in kwargs and kwargs["ids"]:
                    data_info = f", items={len(kwargs['ids'])}"

                # 성능 정보 로깅
                logger.info(f"DB-PERF: {op_key} completed in {elapsed}s{data_info}")

                # 메트릭 저장
                if op_key not in performance_metrics["db_operation_times"]:
                    performance_metrics["db_operation_times"][op_key] = []
                performance_metrics["db_operation_times"][op_key].append(elapsed)

                return result
            except Exception as e:
                # 오류 발생 시간 계산
                elapsed = round(time.time() - start_time, 3)

                # 오류 정보 로깅
                error_type = type(e).__name__
                logger.error(
                    f"DB-ERROR: {op_key} failed after {elapsed}s [error_type={error_type}]"
                )

                # 오류 카운트 증가
                if error_type not in performance_metrics["error_counts"]:
                    performance_metrics["error_counts"][error_type] = 0
                performance_metrics["error_counts"][error_type] += 1

                raise

        return wrapper

    return decorator


def log_embedding_generation(
    field_count: int, vector_size: int, elapsed: float
) -> None:
    """
    임베딩 생성 성능 로깅

    Args:
        field_count: 임베딩 생성에 사용된 필드 수
        vector_size: 생성된 벡터의 크기
        elapsed: 경과 시간(초)
    """
    logger.info(
        f"EMBEDDING: Generated {field_count} fields in {elapsed:.3f}s, vector_size={vector_size}"
    )
    performance_metrics["embedding_generation_times"].append(
        {"field_count": field_count, "vector_size": vector_size, "time": elapsed}
    )


def log_similarity_calculation(
    user_id: str, total_users: int, match_count: int, elapsed: float
) -> None:
    """
    유사도 계산 성능 로깅

    Args:
        user_id: 유사도 계산 대상 사용자 ID
        total_users: 비교 대상 전체 사용자 수
        match_count: 매칭된 사용자 수
        elapsed: 경과 시간(초)
    """
    avg_time_per_user = elapsed / total_users if total_users > 0 else 0
    logger.info(
        f"SIMILARITY: Calculated {match_count} matches out of {total_users} users in {elapsed:.3f}s "
        f"[userId={user_id}, avg_time_per_user={avg_time_per_user:.5f}s]"
    )

    performance_metrics["similarity_calculation_times"].append(
        {
            "user_id": user_id,
            "total_users": total_users,
            "match_count": match_count,
            "time": elapsed,
            "avg_time_per_user": avg_time_per_user,
        }
    )


def log_memory_usage(operation: str = "general") -> None:
    """
    현재 메모리 사용량 로깅

    Args:
        operation: 작업 이름
    """
    memory_mb = _get_memory_usage()
    logger.info(f"MEMORY: {operation} - Current usage: {memory_mb:.2f}MB")
    performance_metrics["memory_usage_samples"].append(memory_mb)


def get_performance_summary() -> Dict[str, Any]:
    """
    누적된 성능 지표 요약 정보 반환
    """
    import statistics

    summary = {}

    # API 응답 시간 요약
    if performance_metrics["api_response_times"]:
        summary["api_response_times"] = {}
        for op_name, times in performance_metrics["api_response_times"].items():
            if times:
                summary["api_response_times"][op_name] = {
                    "count": len(times),
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "p95": (
                        sorted(times)[int(len(times) * 0.95)]
                        if len(times) > 20
                        else max(times)
                    ),
                }

    # 임베딩 생성 시간 요약
    if performance_metrics["embedding_generation_times"]:
        times = [
            item["time"] for item in performance_metrics["embedding_generation_times"]
        ]
        summary["embedding_generation"] = {
            "count": len(times),
            "avg_time": statistics.mean(times) if times else 0,
            "avg_fields": (
                statistics.mean(
                    [
                        item["field_count"]
                        for item in performance_metrics["embedding_generation_times"]
                    ]
                )
                if times
                else 0
            ),
            "avg_vector_size": (
                statistics.mean(
                    [
                        item["vector_size"]
                        for item in performance_metrics["embedding_generation_times"]
                    ]
                )
                if times
                else 0
            ),
        }

    # 유사도 계산 시간 요약
    if performance_metrics["similarity_calculation_times"]:
        times = [
            item["time"] for item in performance_metrics["similarity_calculation_times"]
        ]
        summary["similarity_calculation"] = {
            "count": len(times),
            "avg_time": statistics.mean(times) if times else 0,
            "avg_match_ratio": (
                statistics.mean(
                    [
                        (
                            item["match_count"] / item["total_users"]
                            if item["total_users"] > 0
                            else 0
                        )
                        for item in performance_metrics["similarity_calculation_times"]
                    ]
                )
                if times
                else 0
            ),
        }

    # DB 작업 시간 요약
    if performance_metrics["db_operation_times"]:
        summary["db_operations"] = {}
        for op_key, times in performance_metrics["db_operation_times"].items():
            if times:
                summary["db_operations"][op_key] = {
                    "count": len(times),
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                }

    # 오류 카운트 요약
    if performance_metrics["error_counts"]:
        summary["errors"] = performance_metrics["error_counts"]

    # 메모리 사용량 요약
    if performance_metrics["memory_usage_samples"]:
        summary["memory_usage"] = {
            "samples": len(performance_metrics["memory_usage_samples"]),
            "avg": statistics.mean(performance_metrics["memory_usage_samples"]),
            "max": max(performance_metrics["memory_usage_samples"]),
            "current": (
                performance_metrics["memory_usage_samples"][-1]
                if performance_metrics["memory_usage_samples"]
                else 0
            ),
        }
    # 함수별 메모리 사용량 요약
    if "memory_usage_by_function" in performance_metrics:
        summary["memory_usage_by_function"] = {}
        for func_name, samples in performance_metrics[
            "memory_usage_by_function"
        ].items():
            if samples:
                summary["memory_usage_by_function"][func_name] = {
                    "count": len(samples),
                    "avg": statistics.mean(samples),
                    "max": max(samples),
                    "latest": samples[-1],
                }
    return summary


def reset_performance_metrics() -> None:
    """
    성능 지표 초기화
    """
    for key in performance_metrics:
        if isinstance(performance_metrics[key], dict):
            performance_metrics[key] = {}
        else:
            performance_metrics[key] = []


def _get_memory_usage() -> float:
    """
    현재 프로세스의 메모리 사용량을 MB 단위로 반환
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # MB로 변환
    except Exception:
        return 0.0


def _extract_result_info(result: Any) -> str:
    """
    결과 객체에서 유용한 정보를 추출
    """
    info = ""
    if isinstance(result, dict):
        if "matchedUserCount" in result:
            info += f", matches={result['matchedUserCount']}"
        if "time_taken_seconds" in result:
            info += f", reported_time={result['time_taken_seconds']}s"
        if "updated_similarities" in result:
            info += f", similarities={result['updated_similarities']}"

    return info


def _store_metric(op_name: str, elapsed: float, result: Any = None) -> None:
    """
    성능 지표 저장
    """
    # API 응답 시간 저장
    if op_name not in performance_metrics["api_response_times"]:
        performance_metrics["api_response_times"][op_name] = []
    performance_metrics["api_response_times"][op_name].append(elapsed)

    # 추가 메트릭 (예: 임베딩 또는 유사도 계산 관련)은 전용 함수를 통해 저장
