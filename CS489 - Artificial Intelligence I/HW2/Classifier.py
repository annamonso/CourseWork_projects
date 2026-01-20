import argparse
import pandas as pd
import torch
import transformers
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from evaluate import load as load_metric
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from explainability import load_pipeline, explain_shap, explain_lime
import os
import datetime
import json
import random
from multimodal_model import MultimodalConfig, MultimodalClassifier
from data_collator_mm import DataCollatorWithMeta
from sklearn.preprocessing import StandardScaler
import joblib

######
def build_meta_features(df, n_topics: int = 5):
    # --- Temporal features ---
    dt = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    year = dt.dt.year.fillna(0).astype(int)
    month = dt.dt.month.fillna(0).astype(int)
    dow = dt.dt.dayofweek.fillna(0).astype(int)

    meta = pd.DataFrame({"year": year, "month": month, "dow": dow})

    # --- Content length ---
    meta["content_len"] = df["content"].apply(lambda x: len(str(x).split()))

    # --- Source one-hot encoding (if available) ---
    if "source" in df.columns:
        src_dum = pd.get_dummies(df["source"].fillna("__UNK__"), prefix="src")
        meta = pd.concat([meta, src_dum], axis=1)

    # --- Topic / cluster ID ---
    try:
        sample_texts = df["content"].fillna("").astype(str).tolist()
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        X_tfidf = tfidf.fit_transform(sample_texts)
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        topics = kmeans.fit_predict(X_tfidf)
        meta["topic_id"] = topics
    except Exception as e:
        print("⚠️ Topic extraction skipped:", e)
        meta["topic_id"] = 0

    # Convert categorical topic to one-hot
    meta = pd.get_dummies(meta, columns=["topic_id"], prefix="topic")

    return meta

# ---------------------------
# Utils
# ---------------------------
def get_time_splits_2(data, n_samples, train_split, test_split):
    data = data[['title', 'content', 'label', 'published_utc']].drop_duplicates(subset="content")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if n_samples < len(data):
        data = data.sample(n=n_samples, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(
        data,
        test_size=(1 - train_split),
        stratify=data["label"],
        random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_data_lists(data):
    return (data.title.tolist(), data.content.tolist())

def get_label_list(data):
    return data.label.tolist()

def tokenize_data(titles, content, tokenizer):
    return tokenizer(titles, content, padding=True, truncation=True, max_length=512, return_tensors='pt')

def preprocess_data_2(data, n_samples, train_split, test_split, tokenizer):
    train_df, test_df = get_time_splits_2(data, n_samples, train_split, test_split)
    train_title, train_content = get_data_lists(train_df)
    test_title, test_content = get_data_lists(test_df)
    train_labels = get_label_list(train_df)
    test_labels = get_label_list(test_df)
    train_encodings = tokenize_data(train_title, train_content, tokenizer)
    test_encodings = tokenize_data(test_title, test_content, tokenizer)
    return train_encodings, train_labels, test_encodings, test_labels, train_df, test_df

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, meta_features):
        self.encodings = encodings
        self.labels = labels
        self.meta = meta_features

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "meta_features": torch.tensor(self.meta[idx], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.labels)

def build_meta_features(df):
    # asumimos published_utc presente en df
    dt = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    year = dt.dt.year.fillna(0).astype(int)
    month = dt.dt.month.fillna(0).astype(int)
    day_of_week = dt.dt.dayofweek.fillna(0).astype(int)

    meta = pd.DataFrame({
        "year": year,
        "month": month,
        "dow": day_of_week
    })

    # si existe 'source' en df, puedes añadirlo con one hot
    if "source" in df.columns:
        src_dum = pd.get_dummies(df["source"].fillna("__UNK__"), prefix="src")
        meta = pd.concat([meta, src_dum], axis=1)

    return meta

# ---------------------------
# Main training function (baseline + multimodal)
# ---------------------------
def main(NEWS_PATH='./datasets/articles.pkl',
         MODEL_NAME='roberta-base',
         SAMPLE_SIZE=25000,
         TRAIN_SPLIT=0.75,
         NUM_EPOCHS=4,
         BATCH_SIZE=-1):

    TEST_SPLIT = 1 - TRAIN_SPLIT
    print(NEWS_PATH, MODEL_NAME, SAMPLE_SIZE, TRAIN_SPLIT, TEST_SPLIT, NUM_EPOCHS)

    articles = pd.read_pickle(NEWS_PATH)
    time_sorted = articles.sample(frac=1, random_state=42).reset_index(drop=True)

    models = ['distilbert-base-uncased', 'roberta-base', 'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base']
    chosen_model = MODEL_NAME
    if chosen_model in models:
        model_name = chosen_model
    else:
        try:
            chosen_model = int(chosen_model)
            model_name = models[chosen_model]
        except:
            print("Model not supported.")
            return 0

    learning_dict = dict(zip(models, [5e-5, 3e-5, 3e-5, 3e-5]))
    batch_dict = dict(zip(models, [128, 64, 32, 32]))
    if BATCH_SIZE == -1:
        BATCH_SIZE = batch_dict[model_name]

    print(f"Batch size: {BATCH_SIZE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Using GPU:", torch.cuda.is_available(), "\n")

    # --------------------------------------------------------
    # STEP 1: BASELINE (text only)
    # --------------------------------------------------------
    print("==== BASELINE TRAINING (TEXT ONLY) ====")

    preprocessed_data = preprocess_data_2(
        time_sorted, SAMPLE_SIZE, TRAIN_SPLIT, TEST_SPLIT, tokenizer
    )
    train_encodings, train_labels, test_encodings, test_labels, train_df, test_df = preprocessed_data

    train_dataset_text = torch.utils.data.TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
    )
    test_dataset_text = torch.utils.data.TensorDataset(
        test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels)
    )

    def baseline_collate(batch):
        input_ids, attn_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn_mask),
            "labels": torch.tensor(labels)
        }

    model_baseline = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    acc_metric = load_metric("accuracy")
    auc_metric = load_metric("roc_auc")
    f1_metric = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_scores = softmax(logits.astype('float32'), axis=-1)
        predictions = np.argmax(pred_scores, axis=-1)
        pos_scores = pred_scores[:, 1]
        try:
            auc_value = auc_metric.compute(prediction_scores=pos_scores, references=labels)
        except Exception:
            auc_value = auc_metric.compute(predictions=pos_scores, references=labels)
        return {
            'acc': acc_metric.compute(predictions=predictions, references=labels),
            'auc': auc_value,
            'f1': f1_metric.compute(predictions=predictions, references=labels)
        }

    os.environ["WANDB_DISABLED"] = "true"

    baseline_args = TrainingArguments(
        output_dir="./training_baseline",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=learning_dict[model_name],
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        # evaluation_strategy="epoch"
    )

    trainer_baseline = Trainer(
        model=model_baseline,
        args=baseline_args,
        train_dataset=train_dataset_text,
        eval_dataset=test_dataset_text,
        compute_metrics=compute_metrics,
        data_collator=baseline_collate
    )

    print("Training baseline model...")
    trainer_baseline.train()
    baseline_results = trainer_baseline.evaluate()
    print("Baseline results:", baseline_results)

    # Save baseline metrics
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame([baseline_results]).to_csv("./results/baseline_metrics.csv", index=False)

    # --------------------------------------------------------
    # STEP 2: MULTIMODAL MODEL
    # --------------------------------------------------------
    print("\n==== MULTIMODAL TRAINING (TEXT + METADATA) ====")

    train_meta_df = build_meta_features(train_df)
    test_meta_df = build_meta_features(test_df)

    scaler = StandardScaler()
    train_meta = scaler.fit_transform(train_meta_df.values)
    test_meta = scaler.transform(test_meta_df.reindex(columns=train_meta_df.columns, fill_value=0).values)

    os.makedirs("./artifacts", exist_ok=True)
    joblib.dump(scaler, "./artifacts/meta_scaler.joblib")
    with open("./artifacts/meta_columns.json", "w") as f:
        json.dump(train_meta_df.columns.tolist(), f)

    train_dataset_mm = BuildDataset(train_encodings, train_labels, train_meta)
    test_dataset_mm = BuildDataset(test_encodings, test_labels, test_meta)

    num_meta_features = train_meta.shape[1]
    mm_config = MultimodalConfig(
        base_model_name=model_name,
        num_meta_features=num_meta_features,
        num_labels=2
    )
    model_mm = MultimodalClassifier(mm_config)
    data_collator = DataCollatorWithMeta(tokenizer=tokenizer)

    mm_args = TrainingArguments(
        output_dir="./training_multimodal",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=learning_dict[model_name],
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        # evaluation_strategy="epoch"
    )

    trainer_mm = Trainer(
        model=model_mm,
        args=mm_args,
        train_dataset=train_dataset_mm,
        eval_dataset=test_dataset_mm,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print("Training multimodal model...")
    trainer_mm.train()
    mm_results = trainer_mm.evaluate()
    print("Multimodal results:", mm_results)

    pd.DataFrame([mm_results]).to_csv("./results/multimodal_metrics.csv", index=False)

    print("\n==== COMPARISON ====")
    print(f"Baseline acc: {baseline_results['eval_acc']['accuracy']:.3f} | "
          f"Multimodal acc: {mm_results['eval_acc']['accuracy']:.3f}")
    print(f"Baseline AUC: {baseline_results['eval_auc']['roc_auc']:.3f} | "
          f"Multimodal AUC: {mm_results['eval_auc']['roc_auc']:.3f}")
    print(f"Baseline F1:  {baseline_results['eval_f1']['f1']:.3f} | "
          f"Multimodal F1: {mm_results['eval_f1']['f1']:.3f}")

    print("\nDone ✅")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('news_path', default='./datasets/articles.pkl')
    parser.add_argument('-m', '--model_name', default='roberta-base')
    parser.add_argument('-s', '--sample_size', default=25000, type=int)
    parser.add_argument('-t', '--train_split', default=0.75, type=float)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=-1, type=int)

    args = parser.parse_args()
    main(
        NEWS_PATH=args.news_path,
        MODEL_NAME=args.model_name,
        SAMPLE_SIZE=args.sample_size,
        TRAIN_SPLIT=args.train_split,
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size
    )
