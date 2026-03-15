import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-27B"

if torch.cuda.is_available():
    device="cuda"
elif torch.backends.mps.is_available():
    device= "mps"
else:
    device="cpu"

print("device: ",device)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

print("model ready")

user_input = input("\nUser: ")

prompt = f"User: {user_input}\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Assistant 부분만 출력
answer = response.split("Assistant:")[-1].strip()

print("\nAssistant:", answer)