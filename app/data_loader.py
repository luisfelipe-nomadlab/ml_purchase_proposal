# data_loader.py

import pandas as pd

def ler_dados(caminho: str) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho)
    except Exception as e:
        raise RuntimeError(f"Erro ao ler dados: {e}")