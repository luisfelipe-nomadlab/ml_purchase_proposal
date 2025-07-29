from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import pandas as pd

def avaliar_modelo(modelo, X_test, y_test):
    """
    Calcula métricas de avaliação do modelo.

    Parâmetros:
    - modelo: modelo treinado (ex: DecisionTreeClassifier)
    - X_test: dados de teste (features)
    - y_test: rótulos reais de teste

    Retorna:
    - Um dicionário com as métricas calculadas
    """
    y_pred = modelo.predict(X_test)

    # Se for problema binário e houver probabilidade
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
