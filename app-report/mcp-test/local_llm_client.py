# client.py
import json
import os
import subprocess

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# 0. huggingface 로그인
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_api_key:
    login(token=hf_api_key)
    print("Hugging Face에 성공적으로 로그인했습니다.")
else:
    print("경고: HUGGINGFACE_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 1. LLM 모델 로드
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. MCP 서버 프로세스 시작
server = subprocess.Popen(
    ["python3", "app-report/mcp-test/local_llm_server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdin=subprocess.PIPE,
    text=True,
)


# 3. JSON-RPC 메시지 유틸리티 함수
def create_message(method_name, params, id=None):
    message = {"jsonrpc": "2.0", "method": method_name, "params": params, "id": id}
    return json.dumps(message)


def send_message(message):
    server.stdin.write(message + "\n")
    server.stdin.flush()


def receive_message():
    server_output = json.loads(server.stdout.readline())
    if "result" in server_output:
        return server_output["result"]
    else:
        return "Error"


# 4. 통신 세션 초기화
id = 1
init_message = create_message(
    "initialize",
    {
        "clientInfo": {"name": "Llama Agent", "version": "0.1"},
        "protocolVersion": "2024-11-05",
        "capabilities": {},
    },
    id,
)

send_message(init_message)
response = receive_message()
server_name = response["serverInfo"]["name"]
print("Initializing " + server_name + "...")

init_complete_message = create_message("notifications/initialized", {})
send_message(init_complete_message)
print("Initialization complete.")

# 5. MCP 서버에서 도구 목록 가져오기
id += 1
list_tools_message = create_message("tools/list", {}, id)
send_message(list_tools_message)
response = json.loads(server.stdout.readline())["result"]
print("Available tools:")
for tool in response["tools"]:
    print(f"- {tool['name']}: {tool['description']}")

# 6. Llama 모델용 도구 배열 생성
available_functions = []
for tool in response["tools"]:
    func = {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["inputSchema"]["properties"],
                "required": tool["inputSchema"]["required"],
            },
        },
    }
    available_functions.append(func)

# 7. 모델에 프롬프트와 도구 전달하기
prompt = "What's there in the /tmp directory?"
print(f"\nUser query: {prompt}")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

template = tokenizer.apply_chat_template(
    messages, tools=available_functions, tokenize=False
)

inputs = tokenizer(template, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
generated_text = tokenizer.decode(outputs[0])
print("\nModel response:")
print(generated_text)

# 8. 함수 호출 결과 처리하기
last_line = generated_text.split("\n")[-1]
start_marker = "<|python_tag|>"
end_marker = "<|eom_id|>"
id += 1

if start_marker in last_line and end_marker in last_line:
    code = last_line.split(start_marker)[1].split(end_marker)[0]
    code = json.loads(code)
    print(f"\nExtracted function call: {code}")

    function_call = create_message(
        "tools/call",
        {
            "name": code["function"],
            "arguments": code["parameters"],
        },
        id,
    )

    send_message(function_call)
    response = json.loads(server.stdout.readline())["result"]
    result = response["content"][0]["text"]

    print("\nFunction call result:")
    print(result)
else:
    print("\nNo function call detected in the model's response.")

# 종료 시 서버 프로세스 정리
server.terminate()
