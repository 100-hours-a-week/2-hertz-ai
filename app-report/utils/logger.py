import asyncio
import functools
import logging
import os
import time
from datetime import datetime
from inspect import signature
from typing import Any, Callable, Dict, Optional

import psutil

# --------- 로거 설정 ---------
logger = logging.getLogger("tuning_performance")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(
    f"logs/performance_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)

performance_metrics = {
    "api_response_times": {},
    "error_counts": {},
    "memory_usage_samples": [],
    "memory_usage_by_function": {},
}


# --------- 성능 데코레이터 ---------
def log_performance(
    operation_name: Optional[str] = None,
    include_memory: bool = False,
    include_args: bool = False,
):
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _run_with_logging(
                    func, args, kwargs, operation_name, include_memory, include_args
                )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _run_with_logging_sync(
                    func, args, kwargs, operation_name, include_memory, include_args
                )

            return sync_wrapper

    return decorator


# --------- 내부 실행 래퍼 ---------
def _get_request_info(bound_args) -> tuple[str, str]:
    request_obj = next(
        (arg for arg in bound_args.arguments.values() if hasattr(arg, "category")), None
    )
    category = getattr(request_obj, "category", "unknown") if request_obj else "unknown"
    mbti_pair = (
        f"{getattr(request_obj.userA, 'MBTI', '?')}+{getattr(request_obj.userB, 'MBTI', '?')}"
        if request_obj
        and hasattr(request_obj, "userA")
        and hasattr(request_obj, "userB")
        else "unknown"
    )
    return category, mbti_pair


async def _run_with_logging(
    func, args, kwargs, operation_name, include_memory, include_args
):
    is_async = asyncio.iscoroutinefunction(func)  # ✔️ async 여부 판단
    op_name = operation_name or func.__name__
    bound_args = signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    category, mbti_pair = _get_request_info(bound_args)
    safe_args = _safe_args_for_logging(bound_args.arguments) if include_args else ""

    initial_memory = _get_memory_usage() if include_memory else None
    start_time = time.time()

    try:
        if is_async:
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)

        elapsed = round(time.time() - start_time, 3)
        result_info = _extract_result_info(result)
        memory_info = ""

        if include_memory:
            final_memory = _get_memory_usage()
            memory_diff = final_memory - initial_memory
            memory_info = f"mem_diff={memory_diff:.2f}MB"
            _store_memory_sample(op_name, final_memory)

        log_fields = []
        if include_args and safe_args:
            log_fields.append(safe_args)
        if result_info:
            log_fields.append(result_info.lstrip(", "))
        if memory_info:
            log_fields.append(memory_info)

        log_msg = f"PERF: {op_name} completed in {elapsed}s"
        if log_fields:
            log_msg += " [" + ", ".join(log_fields) + "]"

        logger.info(log_msg)
        _store_metric(op_name, elapsed)
        return result

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        error_type = type(e).__name__
        logger.error(
            f"PERF-ERROR: {op_name} failed after {elapsed}s ["
            f"{safe_args if include_args else ''}"
            f", error_type={error_type}]"
        )
        _record_error_metric(op_name, error_type)
        raise


def _run_with_logging_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    operation_name: Optional[str],
    include_memory: bool,
    include_args: bool,
) -> Any:
    op_name = operation_name or func.__name__
    bound_args = signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    category, mbti_pair = _get_request_info(bound_args)
    safe_args = _safe_args_for_logging(bound_args.arguments) if include_args else ""

    initial_memory = _get_memory_usage() if include_memory else None
    start_time = time.time()

    try:
        result = func(*args, **kwargs)

        elapsed = round(time.time() - start_time, 3)
        result_info = _extract_result_info(result)
        memory_info = ""

        if include_memory:
            final_memory = _get_memory_usage()
            memory_diff = final_memory - initial_memory
            memory_info = f"mem_diff={memory_diff:.2f}MB"
            _store_memory_sample(op_name, final_memory)
        log_fields = []
        if include_args and safe_args:
            log_fields.append(safe_args)
        if result_info:
            log_fields.append(result_info.lstrip(", "))
        if memory_info:
            log_fields.append(memory_info)

        log_msg = f"PERF: {op_name} completed in {elapsed}s"
        if log_fields:
            log_msg += " [" + ", ".join(log_fields) + "]"

        logger.info(log_msg)
        return result

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        error_type = type(e).__name__
        logger.error(
            f"PERF-ERROR: {op_name} failed after {elapsed}s ["
            f"{safe_args if include_args else ''}"
            f"error_type={error_type}]"
        )
        _record_error_metric(op_name, error_type)
        raise


# --------- 유틸 함수 ---------
def _get_memory_usage() -> float:
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _extract_result_info(result: Any) -> str:
    try:
        if hasattr(result, "data") and isinstance(result.data, dict):
            title = result.data.get("title", "")
            content = result.data.get("content", "")
            return f", title_len={len(title)}, content_len={len(content)}"
        elif isinstance(result, dict) and "title" in result and "content" in result:
            return f", title_len={len(result['title'])}, content_len={len(result['content'])}"
    except Exception as e:
        logger.warning(f"[extract_result_info] 실패: {e}")
    return ""


def _safe_args_for_logging(arguments: Dict[str, Any]) -> str:
    try:
        for arg in arguments.values():
            if (
                hasattr(arg, "category")
                and hasattr(arg, "userA")
                and hasattr(arg, "userB")
            ):
                return f"category={arg.category}, mbti_pair={arg.userA.MBTI}+{arg.userB.MBTI}"
    except Exception:
        pass
    return "args_unavailable"


def _store_metric(op_name: str, elapsed: float) -> None:
    if op_name not in performance_metrics["api_response_times"]:
        performance_metrics["api_response_times"][op_name] = []
    performance_metrics["api_response_times"][op_name].append(elapsed)


def _record_error_metric(op_name: str, error_type: str) -> None:
    if op_name not in performance_metrics["error_counts"]:
        performance_metrics["error_counts"][op_name] = {}
    if error_type not in performance_metrics["error_counts"][op_name]:
        performance_metrics["error_counts"][op_name][error_type] = 0
    performance_metrics["error_counts"][op_name][error_type] += 1


def _store_memory_sample(op_name: str, memory_value: float) -> None:
    performance_metrics["memory_usage_samples"].append(memory_value)
    if op_name not in performance_metrics["memory_usage_by_function"]:
        performance_metrics["memory_usage_by_function"][op_name] = []
    performance_metrics["memory_usage_by_function"][op_name].append(memory_value)


# --------- API ---------
def log_memory_usage(operation: str = "general") -> None:
    memory_mb = _get_memory_usage()
    logger.info(f"MEMORY: {operation} - Current usage: {memory_mb:.2f}MB")
    performance_metrics["memory_usage_samples"].append(memory_mb)


def get_performance_summary() -> Dict[str, Any]:
    import statistics

    summary = {}

    if performance_metrics["api_response_times"]:
        summary["api_response_times"] = {
            op: {
                "count": len(times),
                "avg": round(statistics.mean(times), 3),
                "max": round(max(times), 3),
                "min": round(min(times), 3),
            }
            for op, times in performance_metrics["api_response_times"].items()
        }

    if performance_metrics["memory_usage_by_function"]:
        summary["memory_usage_by_function"] = {
            op: {
                "avg": round(statistics.mean(samples), 2),
                "max": round(max(samples), 2),
                "latest": round(samples[-1], 2),
            }
            for op, samples in performance_metrics["memory_usage_by_function"].items()
        }

    if performance_metrics["error_counts"]:
        summary["error_counts"] = performance_metrics["error_counts"]

    return summary


def reset_performance_metrics() -> None:
    for key in performance_metrics:
        performance_metrics[key] = (
            {} if isinstance(performance_metrics[key], dict) else []
        )
