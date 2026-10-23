from __future__ import annotations

import csv
import json
import re

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}


def read_dialogs(path: str) -> list[dict]:
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if not record.get("turns"):
                    continue
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    if not path:
        return {}
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                record_labels = record.get("labels")
                if not record_labels:
                    continue
                label = record_labels.get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def normalize(text: str) -> list[str]:
    words = WORD.findall(text.lower())
    letters = "".join(words)
    language = "russian" if letters and len(CYRILLIC.findall(letters)) > len(letters) / 2 else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    stems = [stemmer.stem(word) for word in words if len(word) >= 2 and word not in stop]
    return stems


def dialog_text(dialog: dict) -> str:
    user_text = " ".join(turn["text"] for turn in dialog["turns"] if turn["role"] == "user")
    return " ".join(normalize(user_text))


def init_reaction_models(train_path: str) -> dict:
    return {"best_reaction_model": joblib.load("artifacts/reaction_model.joblib")}


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    texts = [dialog_text(d) for d in dialogs]
    return [str(x) for x in model.predict(texts)]


def init_topic_models(train_path: str) -> dict:
    return {"best_topic_model": joblib.load("artifacts/topic_model.joblib")}


def predict_topic(model, dialogs: list[dict]) -> list[str]:
    texts = [dialog_text(d) for d in dialogs]
    return [str(x) for x in model.predict(texts)]


def compute_metrics(dialogs, predictions, labels) -> dict[str, float]:
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    if not pairs:
        return {}
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def compute_partition_metrics(dialogs, predictions, labels) -> dict[str, float]:
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    if not pairs:
        return {}
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return {
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def save_predictions(path, dialogs, predictions):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])