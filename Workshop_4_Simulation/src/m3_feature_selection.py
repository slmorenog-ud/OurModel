"""
Workshop 4: Módulo M3 - Feature Selection
Selección de características basada en importancia clínica y ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from config import TARGET_COLUMN, CRITICAL_FEATURES, RANDOM_STATE, ID_COLUMN


def get_clinical_features():
    """
    Retorna las características clínicas críticas identificadas en Workshop 1.
    
    Returns:
        list: Lista de características críticas
    """
    print(f"✓ Características clínicas críticas (Workshop 1): {len(CRITICAL_FEATURES)}")
    for i, feat in enumerate(CRITICAL_FEATURES, 1):
        print(f"   {i}. {feat}")
    
    return CRITICAL_FEATURES.copy()


def calculate_feature_importance(X, y):
    """
    Calcula la importancia de características usando Random Forest.
    
    Args:
        X: DataFrame con features
        y: Serie con target
    
    Returns:
        DataFrame: Importancia de cada feature ordenada
    """
    # Entrenar Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Obtener importancia
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"✓ Importancia calculada para {len(X.columns)} features")
    
    return importance_df


def select_features(df, target_col=TARGET_COLUMN, n_top_features=20):
    """
    Selecciona features combinando criterios clínicos y de ML.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        n_top_features: Número de features top a considerar
    
    Returns:
        tuple: (selected_features, importance_df, X, y)
    """
    # Excluir columnas no-features
    exclude_cols = [ID_COLUMN, target_col, 'efs_time']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Preparar X, y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Manejar valores faltantes en y
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Calcular importancia con ML
    importance_df = calculate_feature_importance(X, y)
    
    # Obtener features clínicas
    clinical_features = get_clinical_features()
    
    # Seleccionar top features por ML
    ml_top_features = importance_df.head(n_top_features)['feature'].tolist()
    
    # Combinar: clínicas + ML (sin duplicados)
    selected_features = []
    
    # Primero agregar clínicas que estén en el dataset
    for feat in clinical_features:
        if feat in X.columns and feat not in selected_features:
            selected_features.append(feat)
    
    # Luego agregar top ML que no estén ya incluidas
    for feat in ml_top_features:
        if feat not in selected_features:
            selected_features.append(feat)
        if len(selected_features) >= n_top_features:
            break
    
    print(f"\n✓ Features seleccionadas: {len(selected_features)}")
    print(f"   - Clínicas presentes: {len([f for f in clinical_features if f in selected_features])}")
    print(f"   - Top ML adicionales: {len(selected_features) - len([f for f in clinical_features if f in selected_features])}")
    
    return selected_features, importance_df, X[selected_features], y


def plot_feature_importance(importance_df, save_path=None, top_n=20):
    """
    Genera visualización de importancia de características.
    
    Args:
        importance_df: DataFrame con importancia de features
        save_path: Ruta para guardar la figura (opcional)
        top_n: Número de features a mostrar
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Seleccionar top features
    plot_df = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    # Crear colores
    colors = sns.color_palette("viridis", len(plot_df))
    
    # Gráfico de barras horizontal
    bars = ax.barh(plot_df['feature'], plot_df['importance'], color=colors)
    
    # Marcar features clínicas
    clinical_features = CRITICAL_FEATURES
    for i, (feat, imp) in enumerate(zip(plot_df['feature'], plot_df['importance'])):
        if feat in clinical_features:
            ax.annotate('★', xy=(imp, i), xytext=(5, 0), 
                       textcoords='offset points', fontsize=12, color='red')
    
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Features por Importancia\n(★ = Feature clínica crítica)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def run_feature_selection(df):
    """
    Pipeline completo del Módulo M3: Feature Selection.
    
    Args:
        df: DataFrame con los datos preprocesados
    
    Returns:
        dict: Resultados de la selección de features
    """
    print("=" * 50)
    print("MÓDULO M3: FEATURE SELECTION")
    print("=" * 50)
    
    # 1. Seleccionar features
    selected_features, importance_df, X, y = select_features(df)
    
    # 2. Preparar resultados
    results = {
        'selected_features': selected_features,
        'importance_df': importance_df,
        'X': X,
        'y': y,
        'n_features': len(selected_features)
    }
    
    print("=" * 50)
    print(f"✓ Selección de features completada: {len(selected_features)} features")
    print("=" * 50)
    
    return results
