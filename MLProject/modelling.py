import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
import os

# =========================
# SETUP DAGSHUB + MLFLOW
# =========================
dagshub.init(
    repo_owner='zulfiana92',
    repo_name='Eksperimen_SML_Zulfiana-Majid',
    mlflow=True
)

mlflow.set_experiment("heart_disease_workflow_ci")

# =========================
# LOAD DATA
# =========================
def load_data():
    train_data = pd.read_csv(
        "MLProject/heart_disease_preprocessing/train_data.csv"
    )
    test_data = pd.read_csv(
        "MLProject/heart_disease_preprocessing/test_data.csv"
    )
    X_train = train_data.drop(columns='target')
    y_train = train_data['target']
    X_test = test_data.drop(columns='target')
    y_test = test_data['target']
    return X_train, X_test, y_train, y_test

# =========================
# ARTEFAK 1: Confusion Matrix
# =========================
def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path

# =========================
# ARTEFAK 2: Feature Importance
# =========================
def save_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45, ha='right'
    )
    plt.tight_layout()
    path = "feature_importance.png"
    plt.savefig(path)
    plt.close()
    return path

# =========================
# ARTEFAK 3: ROC Curve
# =========================
def save_roc_curve(model, X_test, y_test):
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.tight_layout()
    path = "roc_curve.png"
    plt.savefig(path)
    plt.close()
    return path

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="rf_workflow_ci"):
        # Training
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Manual Logging: Parameters
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 2)
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("random_state", 42)

        # Manual Logging: Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        # Manual Logging: Model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Artefak Tambahan
        cm_path = save_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        fi_path = save_feature_importance(model, list(X_train.columns))
        mlflow.log_artifact(fi_path)

        roc_path = save_roc_curve(model, X_test, y_test)
        mlflow.log_artifact(roc_path)

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        print("✅ Accuracy :", accuracy_score(y_test, y_pred))
        print("✅ Precision:", precision_score(y_test, y_pred))
        print("✅ Recall   :", recall_score(y_test, y_pred))
        print("✅ F1 Score :", f1_score(y_test, y_pred))
        print("✅ ROC AUC  :", roc_auc_score(y_test, y_prob))

    print("🎉 Training selesai. Cek MLflow di DagsHub!")
