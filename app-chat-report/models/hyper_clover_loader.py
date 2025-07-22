import os
import re
import time
from threading import Lock
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import logger

load_dotenv(override=True)


# 💡 비속어 패턴 필터 (우회 표현 포함)
BADWORD_PATTERNS = [
    r"\bㅅㅂ\b",
    r"\bㅂㅅ\b",
    r"\bㅗ\b",
    r"\bfuck\b",
    r"\bbitch\b",
    r"\basshole\b",
]


def matches_badword(text: str) -> bool:
    for pattern in BADWORD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


class HyperCLOVAXLoader:
    def __init__(
        self, model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    ):
        self.model_name = model_name
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(2)
        self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        start_time = time.time()
        logger.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HUGGINGFACE_TOKEN")
        )

        logger.info(f"Loading model from {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        ).to(self.device)

        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 16,
            "do_sample": False,
        }

        load_time = time.time() - start_time
        logger.info(f"Model loading completed in {load_time:.2f} seconds")

    def classify(self, text: str) -> Dict[str, str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 사용자의 메시지가 유해한지 판단하는 인공지능입니다. "
                    "유해한 메시지에는 욕설, 비방, 차별, 성적 내용, 폭력적인 표현이 포함될 수 있습니다. "
                    "예: 시11발, ㅅㅂ, ㅂㅅ, ㅗ 등의 표현도 유해로 간주해야 합니다. "
                    "유해한 경우에는 '유해'라고만 답하고, 아니라면 '안전'이라고만 답하세요."
                ),
            },
            {
                "role": "user",
                "content": f'메시지: "{text}"\n\n이 메시지는 유해한가요?',
            },
        ]
        response = self.get_response(messages)
        result_text = response.get("content", "").lower()

        is_model_toxic = any(
            keyword in result_text for keyword in ["유해", "제재", "부적절"]
        )
        is_pattern_toxic = matches_badword(text)

        label = "toxic" if (is_model_toxic or is_pattern_toxic) else "safe"

        return {"label": label, "raw": result_text}

    def get_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        start_time = time.time()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_config)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            end_time = time.time()
            logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")
            return {
                "status_code": 200,
                "content": generated_text.strip(),
                "inference_time": end_time - start_time,
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"status_code": 500, "error": str(e)}


class ModelSingleton:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    model_name = os.getenv(
                        "MODEL_NAME",
                        "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                    )
                    logger.info(
                        f"Initializing HyperCLOVAXLoader with model: {model_name}"
                    )
                    cls._instance = HyperCLOVAXLoader(model_name)
        return cls._instance
