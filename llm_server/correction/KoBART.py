from transformers import AutoTokenizer, BartForConditionalGeneration
from make_dataset import dataset
model_name = "gogamza/kobart-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True
)

print(tokenized_datasets["train"][0].keys())
print(tokenized_datasets["train"][0])