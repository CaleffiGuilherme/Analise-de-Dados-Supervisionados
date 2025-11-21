import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def load_and_process_data():
    print("[INFO] Baixando dataset via ucimlrepo (ID=320)...")
    
    student_performance = fetch_ucirepo(id=320)
    
    X_raw = student_performance.data.features
    y_raw = student_performance.data.targets
    
    df = pd.concat([X_raw, y_raw], axis=1)

    print(f"[INFO] Dados brutos carregados. Linhas: {df.shape[0]}")

    df = df.dropna(subset=['G3'])
    df['target'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    colunas_para_remover = ['G1', 'G2', 'G3']
    
    cols_to_drop = [c for c in colunas_para_remover if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('target', axis=1).values
    y = df['target'].values

    print("[INFO] Dividindo dados em Treino e Teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"[INFO] Processamento concluído. Features finais: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_process_data()