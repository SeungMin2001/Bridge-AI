from transformers import AutoTokenizer, BartForConditionalGeneration

model_name = "gogamza/kobart-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)