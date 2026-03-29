import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration


# =========================
# 1. 경로 설정
# =========================
LOG_CSV_PATH = r"C:\Users\user\Documents\last_project\models\kobart_correction_run\log_history.csv"
GRAPH_SAVE_DIR = r"C:\Users\user\Documents\last_project\models\kobart_analysis"

BASE_MODEL_NAME = "gogamza/kobart-base-v2"
FINETUNED_MODEL_PATH = r"C:\Users\user\Documents\last_project\models\kobart_correction"

TEST_CSV_PATH = r"C:\Users\user\Documents\last_project\data\pair_dataset\test_pairs.csv"

MAX_INPUT_LENGTH = 128
MAX_GENERATION_LENGTH = 128
BATCH_SIZE = 8

os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)


# =========================
# 2. 로그 기반 그래프 저장
# =========================
def save_training_graphs(log_csv_path, save_dir):
    df = pd.read_csv(log_csv_path)

    # train loss
    train_df = df[df["loss"].notna()].copy() if "loss" in df.columns else pd.DataFrame()

    if not train_df.empty and "step" in train_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(train_df["step"], train_df["loss"])
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Train Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "train_loss.png"))
        plt.close()

    # validation loss / CER
    eval_df = df[df["eval_loss"].notna()].copy() if "eval_loss" in df.columns else pd.DataFrame()

    if not eval_df.empty:
        if "epoch" in eval_df.columns:
            x = eval_df["epoch"]
            x_label = "Epoch"
        else:
            x = range(len(eval_df))
            x_label = "Evaluation Step"

        # validation loss
        plt.figure(figsize=(8, 5))
        plt.plot(x, eval_df["eval_loss"])
        plt.xlabel(x_label)
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "validation_loss.png"))
        plt.close()

        # validation CER
        if "eval_cer" in eval_df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(x, eval_df["eval_cer"])
            plt.xlabel(x_label)
            plt.ylabel("Validation CER")
            plt.title("Validation CER")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "validation_cer.png"))
            plt.close()

        # validation exact match
        if "eval_exact_match" in eval_df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(x, eval_df["eval_exact_match"])
            plt.xlabel(x_label)
            plt.ylabel("Validation Exact Match")
            plt.title("Validation Exact Match")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "validation_exact_match.png"))
            plt.close()


# =========================
# 3. test 데이터 로드
# =========================
def load_test_texts(test_csv_path):
    df = pd.read_csv(test_csv_path)
    inputs = df["input_text"].astype(str).tolist()
    targets = df["target_text"].astype(str).tolist()
    return df, inputs, targets


# =========================
# 4. 모델 로드
# =========================
def load_model_and_tokenizer(model_path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = BartForConditionalGeneration.from_pretrained(model_path_or_name)
    return tokenizer, model


# =========================
# 5. 배치 추론
# =========================
def generate_predictions(model, tokenizer, texts, batch_size=8):
    model.eval()
    all_preds = []

    # CUDA 있으면 자동 사용
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception:
        device = "cpu"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )

        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=MAX_GENERATION_LENGTH,
            num_beams=4
        )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        all_preds.extend(preds)

    return all_preds


# =========================
# 6. CER 계산
# =========================
def compute_cer(predictions, references):
    cer_metric = evaluate.load("cer")
    return cer_metric.compute(predictions=predictions, references=references)


# =========================
# 7. 비교표 저장
# =========================
def save_comparison_table(
    original_df,
    base_preds,
    finetuned_preds,
    save_dir
):
    result_df = original_df.copy()
    result_df["base_prediction"] = base_preds
    result_df["finetuned_prediction"] = finetuned_preds

    save_path = os.path.join(save_dir, "test_predictions_comparison.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return save_path


# =========================
# 8. 성능 요약 저장
# =========================
def save_metrics_summary(inputs, targets, base_preds, finetuned_preds, save_dir):
    whisper_cer = compute_cer(inputs, targets)
    base_cer = compute_cer(base_preds, targets)
    finetuned_cer = compute_cer(finetuned_preds, targets)

    summary = {
        "whisper_input_cer": whisper_cer,
        "base_kobart_cer": base_cer,
        "finetuned_kobart_cer": finetuned_cer
    }

    json_path = os.path.join(save_dir, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame([summary])
    csv_path = os.path.join(save_dir, "metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return summary, json_path, csv_path


# =========================
# 9. 실행
# =========================
def main():
    # 1) 그래프 저장
    save_training_graphs(LOG_CSV_PATH, GRAPH_SAVE_DIR)

    # 2) test 데이터 준비
    test_df, test_inputs, test_targets = load_test_texts(TEST_CSV_PATH)

    # 3) base KoBART 추론
    base_tokenizer, base_model = load_model_and_tokenizer(BASE_MODEL_NAME)
    base_preds = generate_predictions(
        base_model,
        base_tokenizer,
        test_inputs,
        batch_size=BATCH_SIZE
    )

    # 4) fine-tuned KoBART 추론
    ft_tokenizer, ft_model = load_model_and_tokenizer(FINETUNED_MODEL_PATH)
    finetuned_preds = generate_predictions(
        ft_model,
        ft_tokenizer,
        test_inputs,
        batch_size=BATCH_SIZE
    )

    # 5) 비교표 저장
    comparison_csv_path = save_comparison_table(
        test_df,
        base_preds,
        finetuned_preds,
        GRAPH_SAVE_DIR
    )

    # 6) CER 요약 저장
    summary, json_path, csv_path = save_metrics_summary(
        test_inputs,
        test_targets,
        base_preds,
        finetuned_preds,
        GRAPH_SAVE_DIR
    )

    print("분석 완료")
    print(f"그래프 저장 폴더: {GRAPH_SAVE_DIR}")
    print(f"비교표 저장: {comparison_csv_path}")
    print(f"지표 JSON 저장: {json_path}")
    print(f"지표 CSV 저장: {csv_path}")
    print("성능 요약:", summary)


if __name__ == "__main__":
    main()