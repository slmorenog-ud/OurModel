"""
Workshop 4: Módulo M1 - Data Preprocessing
Preprocesamiento de datos para el sistema de predicción de supervivencia post-HCT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import TARGET_COLUMN, MISSING_THRESHOLD, ID_COLUMN


def load_data(train_path, dict_path=None):
    """
    Carga el dataset de entrenamiento y opcionalmente el diccionario de datos.
    
    Args:
        train_path: Ruta al archivo CSV de entrenamiento
        dict_path: Ruta al diccionario de datos (opcional)
    
    Returns:
        tuple: (df_train, df_dict) o (df_train, None)
    """
    df_train = pd.read_csv(train_path)
    df_dict = pd.read_csv(dict_path) if dict_path else None
    
    print(f"✓ Dataset cargado: {df_train.shape[0]} filas, {df_train.shape[1]} columnas")
    
    return df_train, df_dict


def get_feature_types(df, df_dict=None):
    """
    Identifica columnas numéricas y categóricas.
    
    Args:
        df: DataFrame con los datos
        df_dict: Diccionario de datos (opcional)
    
    Returns:
        tuple: (numeric_features, categorical_features)
    """
    # Excluir columnas de ID y target
    exclude_cols = [ID_COLUMN, TARGET_COLUMN, 'efs_time']
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in exclude_cols]
    
    print(f"✓ Features numéricas: {len(numeric_features)}")
    print(f"✓ Features categóricas: {len(categorical_features)}")
    
    return numeric_features, categorical_features


def remove_high_missing_columns(df, numeric_features, categorical_features, threshold=MISSING_THRESHOLD):
    """
    Elimina columnas con alto porcentaje de valores faltantes.
    
    Args:
        df: DataFrame con los datos
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
        threshold: Umbral máximo de valores faltantes (default: 0.30)
    
    Returns:
        tuple: (df_filtered, numeric_filtered, categorical_filtered)
    """
    # Calcular porcentaje de missing
    missing_pct = df.isnull().sum() / len(df)
    
    # Identificar columnas a eliminar
    cols_to_remove = missing_pct[missing_pct > threshold].index.tolist()
    
    # Filtrar features
    numeric_filtered = [col for col in numeric_features if col not in cols_to_remove]
    categorical_filtered = [col for col in categorical_features if col not in cols_to_remove]
    
    # No eliminar columnas del DataFrame aún, solo filtrar las listas
    print(f"✓ Columnas con >{threshold*100:.0f}% missing removidas: {len(cols_to_remove)}")
    
    return df, numeric_filtered, categorical_filtered


def impute_missing_values(df, numeric_features, categorical_features):
    """
    Imputa valores faltantes: mediana para numéricas, moda para categóricas.
    
    Args:
        df: DataFrame con los datos
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
    
    Returns:
        DataFrame: DataFrame con valores imputados
    """
    df_imputed = df.copy()
    
    # Imputar numéricas con mediana
    for col in numeric_features:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
    
    # Imputar categóricas con moda
    for col in categorical_features:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            mode_val = df_imputed[col].mode()
            if len(mode_val) > 0:
                df_imputed[col].fillna(mode_val[0], inplace=True)
    
    print(f"✓ Valores imputados - Numéricas: {len(numeric_features)}, Categóricas: {len(categorical_features)}")
    
    return df_imputed


def encode_target(df):
    """
    Codifica la variable target: 'Event'=1, 'Censoring'=0.
    
    Args:
        df: DataFrame con los datos
    
    Returns:
        DataFrame: DataFrame con target codificado
    """
    df_encoded = df.copy()
    
    if TARGET_COLUMN in df_encoded.columns:
        df_encoded[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map({
            'Event': 1,
            'Censoring': 0
        })
        print(f"✓ Target codificado: Event=1, Censoring=0")
    
    return df_encoded


def encode_categorical_features(df, categorical_features):
    """
    Codifica features categóricas usando LabelEncoder.
    
    Args:
        df: DataFrame con los datos
        categorical_features: Lista de features categóricas
    
    Returns:
        tuple: (DataFrame codificado, dict de encoders)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_features:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Manejar valores nulos convirtiéndolos a string
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    print(f"✓ Features categóricas codificadas: {len(categorical_features)}")
    
    return df_encoded, encoders


def normalize_numeric_features(df, numeric_features):
    """
    Normaliza features numéricas usando StandardScaler.
    
    Args:
        df: DataFrame con los datos
        numeric_features: Lista de features numéricas
    
    Returns:
        tuple: (DataFrame normalizado, StandardScaler)
    """
    df_normalized = df.copy()
    scaler = StandardScaler()
    
    cols_to_scale = [col for col in numeric_features if col in df_normalized.columns]
    
    if cols_to_scale:
        df_normalized[cols_to_scale] = scaler.fit_transform(df_normalized[cols_to_scale])
        print(f"✓ Features numéricas normalizadas: {len(cols_to_scale)}")
    
    return df_normalized, scaler


def preprocess_pipeline(train_path, dict_path=None):
    """
    Pipeline completo de preprocesamiento M1.
    
    Args:
        train_path: Ruta al archivo CSV de entrenamiento
        dict_path: Ruta al diccionario de datos (opcional)
    
    Returns:
        tuple: (df_processed, numeric_features, categorical_features, encoders, scaler)
    """
    print("=" * 50)
    print("MÓDULO M1: DATA PREPROCESSING")
    print("=" * 50)
    
    # 1. Cargar datos
    df, df_dict = load_data(train_path, dict_path)
    
    # 2. Identificar tipos de features
    numeric_features, categorical_features = get_feature_types(df, df_dict)
    
    # 3. Remover columnas con alto missing
    df, numeric_features, categorical_features = remove_high_missing_columns(
        df, numeric_features, categorical_features, MISSING_THRESHOLD
    )
    
    # 4. Codificar target
    df = encode_target(df)
    
    # 5. Imputar valores faltantes
    df = impute_missing_values(df, numeric_features, categorical_features)
    
    # 6. Codificar categóricas
    df, encoders = encode_categorical_features(df, categorical_features)
    
    # 7. Normalizar numéricas
    df, scaler = normalize_numeric_features(df, numeric_features)
    
    print("=" * 50)
    print(f"✓ Preprocesamiento completado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("=" * 50)
    
    return df, numeric_features, categorical_features, encoders, scaler
