import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_NAME = "gogamza/kobart-base-v2"
OUTPUT_DIR = "./kobart_run1"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

dataset = load_dataset(
    "csv",
    data_files={
        "train": "C:/Users/user/Documents/ksponspeech_data/train_pairs.csv",
        "validation": "C:/Users/user/Documents/ksponspeech_data/valid_pairs.csv",
    }
)

print(dataset["train"].column_names)
print(dataset["train"][0])

# 예시: 실제 컬럼명에 맞게 수정
INPUT_COL = "input_text"
TARGET_COL = "target_text"

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = ["" if x is None else str(x) for x in examples[INPUT_COL]]
    targets = ["" if x is None else str(x) for x in examples[TARGET_COL]]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./kobart_run1",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,

    evaluation_strategy="epoch",   # 또는 네 버전이면 eval_strategy="epoch"
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,

    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,   # 네 버전에서 안 되면 이 줄 제거
)

train_result = trainer.train()

# 1) 최종 모델 저장
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# 2) best model도 별도 저장
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
trainer.model.save_pretrained(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

# 3) train/eval metrics 저장
metrics = train_result.metrics
trainer.save_metrics("train", metrics)
trainer.save_state()

eval_metrics = trainer.evaluate()
trainer.save_metrics("eval", eval_metrics)

# 4) 학습 인자 저장
with open(os.path.join(OUTPUT_DIR, "training_args.json"), "w", encoding="utf-8") as f:
    json.dump(training_args.to_dict(), f, ensure_ascii=False, indent=2)

# 5) log history 저장
log_history = trainer.state.log_history
log_df = pd.DataFrame(log_history)
log_csv_path = os.path.join(OUTPUT_DIR, "log_history.csv")
log_df.to_csv(log_csv_path, index=False, encoding="utf-8-sig")

# 6) train loss / eval loss 그래프 저장
if "loss" in log_df.columns:
    train_loss_df = log_df[log_df["loss"].notna() & log_df["step"].notna()]
    if len(train_loss_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_df["step"], train_loss_df["loss"])
        plt.xlabel("step")
        plt.ylabel("train loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"))
        plt.close()

if "eval_loss" in log_df.columns:
    eval_loss_df = log_df[log_df["eval_loss"].notna() & log_df["step"].notna()]
    if len(eval_loss_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_loss_df["step"], eval_loss_df["eval_loss"])
        plt.xlabel("step")
        plt.ylabel("eval loss")
        plt.title("Validation Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "valid_loss.png"))
        plt.close()

print("학습 완료")
print("최종 모델 저장:", FINAL_MODEL_DIR)
print("최고 성능 모델 저장:", BEST_MODEL_DIR)
print("로그 CSV 저장:", log_csv_path)
print("그래프 저장:", os.path.join(OUTPUT_DIR, "train_loss.png"))
print("그래프 저장:", os.path.join(OUTPUT_DIR, "valid_loss.png"))