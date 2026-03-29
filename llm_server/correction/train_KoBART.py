import os
import json
import csv
import numpy as np
import evaluate

from transformers import AutoTokenizer, BartForConditionalGeneration
from make_dataset import dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


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

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# CER metric
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # labels의 -100은 decode 전에 pad_token_id로 바꿈
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_texts = [x.strip() for x in pred_texts]
    label_texts = [x.strip() for x in label_texts]

    cer = cer_metric.compute(predictions=pred_texts, references=label_texts)
    exact_match = np.mean([p == l for p, l in zip(pred_texts, label_texts)])

    return {
        "cer": cer,
        "exact_match": float(exact_match)
    }


output_dir = r"C:\Users\user\Documents\last_project\models\kobart_correction_run"
final_save_path = r"C:\Users\user\Documents\last_project\models\kobart_correction"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,                 # 체크포인트 2개만 유지
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    load_best_model_at_end=True,        # 가장 좋은 validation 성능 모델 자동 로드
    metric_for_best_model="cer",        # CER 기준
    greater_is_better=False,            # CER은 낮을수록 좋음
    report_to="none",
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 최종 모델 저장
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

# 최종 validation 결과 1회 저장
final_eval = trainer.evaluate()
print(final_eval)

# 로그 저장
log_history = trainer.state.log_history

os.makedirs(output_dir, exist_ok=True)

json_path = os.path.join(output_dir, "log_history.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(log_history, f, ensure_ascii=False, indent=2)

# csv 저장
csv_path = os.path.join(output_dir, "log_history.csv")
if len(log_history) > 0:
    fieldnames = sorted(set().union(*[d.keys() for d in log_history]))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_history)

# 최종 evaluation 결과 저장
eval_path = os.path.join(output_dir, "final_eval.json")
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(final_eval, f, ensure_ascii=False, indent=2)

print("학습 완료")
print(f"최종 모델 저장 경로: {final_save_path}")
print(f"로그 저장 경로: {json_path}")
print(f"CSV 로그 저장 경로: {csv_path}")
print(f"최종 eval 저장 경로: {eval_path}")