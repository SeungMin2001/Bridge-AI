from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3.5-27B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate(prompt, max_tokens=300):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def search_agent(query):

    prompt = f"""
You are a research search agent.

User question:
{query}

List key information needed to answer the question.
Provide bullet points of useful facts to search.
"""

    return generate(prompt)


def report_agent(query, research):

    prompt = f"""
You are a report writing agent.

User question:
{query}

Research notes:
{research}

Write a short structured report with:

- introduction
- key points
- conclusion
"""

    return generate(prompt)
