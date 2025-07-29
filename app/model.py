import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from app.config import TEST_SIZE, RANDOM_STATE

def treinar_modelo(df: pd.DataFrame, target: str = "Purchased"):
    X = df.drop(columns=[target])
    y = df[target]

    n_bootstrap = 100
    metrics = []

    for i in range(n_bootstrap):
        # Amostragem com reposição
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=TEST_SIZE, stratify=y_sample, random_state=RANDOM_STATE
        )

        # Treinamento do modelo
        modelo = DecisionTreeClassifier(
            max_depth=5,
            max_leaf_nodes=None,
            min_samples_leaf=10,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        modelo.fit(X_train, y_train)

    return modelo, X_test, y_test

