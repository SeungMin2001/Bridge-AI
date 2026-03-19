import os
import numpy as np
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = "gogamza/kobart-base-v2"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128

# 1. 데이터 로드
# 예: train.jsonl, valid.jsonl
# 각 줄은 {"source": "...", "target": "..."} 형태
dataset = load_dataset(
    "json",
    data_files={
        "train": "C:/Users/user/Documents/ksponspeech_data/train_pairs.csv",
        "validation": "C:/Users/user/Documents/ksponspeech_data/valid_pairs.csv",
    }
)

# 2. 토크나이저 / 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 3. 전처리
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 4. data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# 5. 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./kobart-corrector",
    overwrite_output_dir=True,

    # 학습 관련
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,

    # 평가 / 저장
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # 생성 기반 평가/예측
    predict_with_generate=True,

    # mixed precision
    fp16=True,   # CUDA일 때 사용
    # bf16=True, # bf16 지원 GPU면 대안

    report_to="none"
)

# 6. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. 학습
trainer.train()

# 8. 저장
trainer.save_model("./kobart-corrector-best")
tokenizer.save_pretrained("./kobart-corrector-best")