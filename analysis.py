import json
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from solutions_utils import dialog_text, normalize, read_dialogs as _read_dialogs_raw

os.makedirs("artifacts", exist_ok=True)


def read_dialogs(path):
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if not record.get("turns"):
                    continue
                dialogs.append(record)
    return dialogs


def get_topic_for_dialog(dialog, top_topics):
    labels = dialog.get("labels")
    if not labels:
        return None
    user_turns = labels.get("user_turns")
    if not user_turns:
        return None
    task_types = [t["task_type"] for t in user_turns if "task_type" in t]
    if not task_types:
        return None
    valid = [tt for tt in task_types if tt in top_topics]
    if not valid:
        return "other"
    return Counter(valid).most_common(1)[0][0]


def make_models():
    return {
        "dummy_most_frequent": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "dummy_stratified": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", DummyClassifier(strategy="stratified", random_state=42)),
        ]),
        "multinomial_nb": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", MultinomialNB()),
        ]),
        "linear_svc": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LinearSVC(random_state=42, max_iter=5000)),
        ]),
        "logistic_c01": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(C=0.1, random_state=42, max_iter=1000)),
        ]),
    }


def train_and_pick(X, y, task_name, errors_path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n--- {task_name} ---")
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"train распределение: {Counter(y_train)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    cv_scores = {}
    best_name, best_score = None, -1.0
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_f1 = float(np.mean(scores))
        cv_scores[name] = mean_f1
        print(f"  CV {name}: F1-macro = {mean_f1:.4f}")
        if mean_f1 > best_score:
            best_score, best_name = mean_f1, name

    print(f"Лучшая модель: {best_name} (F1-macro={best_score:.4f})")
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Тест accuracy: {acc:.4f}")
    print(f"Тест F1-macro: {f1m:.4f}")
    print("classification_report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("confusion_matrix (labels по алфавиту):")
    print(confusion_matrix(y_test, y_pred))

    # Дамп ошибок для раздела "Разбор ошибок"
    try:
        import pandas as pd
        err_rows = []
        for text, truth, pred in zip(X_test, y_test, y_pred):
            if truth != pred:
                err_rows.append({"text": text, "y_true": truth, "y_pred": pred})
        pd.DataFrame(err_rows).to_csv(errors_path, index=False)
        print(f"Ошибки сохранены: {errors_path} ({len(err_rows)} шт.)")
    except ImportError:
        pass

    return best_model, best_name, cv_scores, acc, f1m


def main():
    all_dialogs = read_dialogs("hw1_train.jsonl")
    print(f"Загружено диалогов: {len(all_dialogs)}")

    texts = [dialog_text(d) for d in all_dialogs]

    # --- Задача 1: реакция ---
    idx_r = [i for i, d in enumerate(all_dialogs) if d.get("labels") and "reaction" in d["labels"]]
    X_r = [texts[i] for i in idx_r]
    y_r = [all_dialogs[i]["labels"]["reaction"] for i in idx_r]

    best_r, name_r, cv_r, acc_r, f1_r = train_and_pick(
        X_r, y_r, "Задача 1: реакция", "artifacts/errors_reaction.csv"
    )
    joblib.dump(best_r, "artifacts/reaction_model.joblib")
    print(f"Сохранено: artifacts/reaction_model.joblib ({name_r})")

    # --- Задача 2: темы ---
    all_task_types = []
    for d in all_dialogs:
        labels = d.get("labels")
        if not labels:
            continue
        user_turns = labels.get("user_turns")
        if not user_turns:
            continue
        for turn in user_turns:
            if "task_type" in turn:
                all_task_types.append(turn["task_type"])

    top_task_types = [tt for tt, _ in Counter(all_task_types).most_common(5)]
    print(f"\nТемы (top-5 task_type): {top_task_types}")

    X_t, y_t = [], []
    for i, d in enumerate(all_dialogs):
        topic = get_topic_for_dialog(d, top_task_types)
        if topic is not None:
            X_t.append(texts[i])
            y_t.append(topic)

    print(f"Распределение тем в обучении: {Counter(y_t)}")

    best_t, name_t, cv_t, acc_t, f1_t = train_and_pick(
        X_t, y_t, "Задача 2: темы", "artifacts/errors_topic.csv"
    )
    joblib.dump(best_t, "artifacts/topic_model.joblib")
    joblib.dump(top_task_types, "artifacts/topics.joblib")
    print(f"Сохранено: artifacts/topic_model.joblib ({name_t})")
    print("Сохранено: artifacts/topics.joblib")

    # --- Сводка для README ---
    print("\n=== Сводка для README ===")
    print("Reaction CV:", {k: round(v, 4) for k, v in cv_r.items()})
    print(f"Reaction best={name_r}, test_acc={acc_r:.4f}, test_f1={f1_r:.4f}")
    print("Topic CV:", {k: round(v, 4) for k, v in cv_t.items()})
    print(f"Topic best={name_t}, test_acc={acc_t:.4f}, test_f1={f1_t:.4f}")
    print("Top topics:", top_task_types)


if __name__ == "__main__":
    main()