"""
Workshop 4: Módulo M5 - Fairness Calibration
Calibración de equidad para reducir sesgo en predicciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from config import TARGET_COLUMN, EQUITY_COLUMN, BIAS_THRESHOLD


def calculate_fairness_metrics(y_true, y_pred, groups):
    """
    Calcula métricas de equidad por grupo demográfico.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        groups: Grupos demográficos
    
    Returns:
        DataFrame: Métricas de equidad por grupo
    """
    df_eval = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': groups
    })
    
    metrics = []
    
    for group in df_eval['group'].unique():
        group_data = df_eval[df_eval['group'] == group]
        
        # Calcular métricas
        true_positive_rate = (
            (group_data['y_true'] == 1) & (group_data['y_pred'] == 1)
        ).sum() / max((group_data['y_true'] == 1).sum(), 1)
        
        false_positive_rate = (
            (group_data['y_true'] == 0) & (group_data['y_pred'] == 1)
        ).sum() / max((group_data['y_true'] == 0).sum(), 1)
        
        positive_pred_rate = group_data['y_pred'].mean()
        
        metrics.append({
            'group': group,
            'n_samples': len(group_data),
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'positive_pred_rate': positive_pred_rate,
            'base_rate': group_data['y_true'].mean()
        })
    
    return pd.DataFrame(metrics)


def calculate_disparity(fairness_df, metric='positive_pred_rate'):
    """
    Calcula la disparidad máxima entre grupos.
    
    Args:
        fairness_df: DataFrame con métricas de equidad
        metric: Métrica a usar para calcular disparidad
    
    Returns:
        dict: Información de disparidad
    """
    max_val = fairness_df[metric].max()
    min_val = fairness_df[metric].min()
    disparity = max_val - min_val
    
    return {
        'metric': metric,
        'max_value': max_val,
        'min_value': min_val,
        'max_group': fairness_df.loc[fairness_df[metric].idxmax(), 'group'],
        'min_group': fairness_df.loc[fairness_df[metric].idxmin(), 'group'],
        'disparity': disparity,
        'threshold_met': disparity <= BIAS_THRESHOLD
    }


def apply_threshold_calibration(y_proba, groups, target_rate=None):
    """
    Aplica calibración de umbral por grupo para reducir disparidad.
    
    Args:
        y_proba: Probabilidades predichas
        groups: Grupos demográficos
        target_rate: Tasa objetivo (si None, usa la media global)
    
    Returns:
        tuple: (y_pred_calibrated, thresholds_by_group)
    """
    df_calib = pd.DataFrame({
        'y_proba': y_proba,
        'group': groups
    })
    
    if target_rate is None:
        target_rate = y_proba.mean()
    
    thresholds = {}
    y_pred_calibrated = np.zeros(len(y_proba))
    
    for group in df_calib['group'].unique():
        group_mask = df_calib['group'] == group
        group_proba = df_calib.loc[group_mask, 'y_proba']
        
        # Encontrar umbral que produce la tasa objetivo
        sorted_proba = np.sort(group_proba)[::-1]
        target_idx = int(len(sorted_proba) * target_rate)
        target_idx = min(target_idx, len(sorted_proba) - 1)
        threshold = sorted_proba[target_idx]
        
        thresholds[group] = threshold
        y_pred_calibrated[group_mask] = (group_proba >= threshold).astype(int)
    
    print(f"✓ Calibración de umbrales aplicada:")
    for group, thresh in thresholds.items():
        print(f"   {group}: umbral = {thresh:.4f}")
    
    return y_pred_calibrated, thresholds


def plot_fairness_metrics(fairness_df, save_path=None):
    """
    Genera visualización de métricas de equidad.
    
    Args:
        fairness_df: DataFrame con métricas de equidad
        save_path: Ruta para guardar la figura
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = sns.color_palette("husl", len(fairness_df))
    
    # Gráfico 1: True Positive Rate
    ax1 = axes[0]
    bars1 = ax1.bar(fairness_df['group'], fairness_df['true_positive_rate'], color=colors)
    ax1.axhline(y=fairness_df['true_positive_rate'].mean(), color='red', linestyle='--',
                label='Media')
    ax1.set_xlabel('Grupo')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Tasa de Verdaderos Positivos por Grupo')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Gráfico 2: False Positive Rate
    ax2 = axes[1]
    bars2 = ax2.bar(fairness_df['group'], fairness_df['false_positive_rate'], color=colors)
    ax2.axhline(y=fairness_df['false_positive_rate'].mean(), color='red', linestyle='--',
                label='Media')
    ax2.axhline(y=BIAS_THRESHOLD, color='green', linestyle=':', label=f'Umbral: {BIAS_THRESHOLD}')
    ax2.set_xlabel('Grupo')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_title('Tasa de Falsos Positivos por Grupo')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Gráfico 3: Positive Prediction Rate
    ax3 = axes[2]
    bars3 = ax3.bar(fairness_df['group'], fairness_df['positive_pred_rate'], color=colors)
    ax3.bar(fairness_df['group'], fairness_df['base_rate'], alpha=0.3, color='gray',
            label='Base Rate')
    ax3.set_xlabel('Grupo')
    ax3.set_ylabel('Tasa')
    ax3.set_title('Tasa de Predicciones Positivas vs Base Rate')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def run_fairness_calibration(y_true, y_pred, y_proba, groups):
    """
    Pipeline completo del Módulo M5: Fairness Calibration.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones originales
        y_proba: Probabilidades predichas
        groups: Grupos demográficos
    
    Returns:
        dict: Resultados de calibración de equidad
    """
    print("=" * 50)
    print("MÓDULO M5: FAIRNESS CALIBRATION")
    print("=" * 50)
    
    # 1. Calcular métricas antes de calibración
    print("\n>>> Métricas antes de calibración:")
    fairness_before = calculate_fairness_metrics(y_true, y_pred, groups)
    print(fairness_before.to_string(index=False))
    
    disparity_before = calculate_disparity(fairness_before)
    print(f"\n   Disparidad en predicciones positivas: {disparity_before['disparity']:.4f}")
    print(f"   Umbral cumplido: {disparity_before['threshold_met']}")
    
    # 2. Aplicar calibración
    print("\n>>> Aplicando calibración de umbrales...")
    y_pred_calibrated, thresholds = apply_threshold_calibration(y_proba, groups)
    
    # 3. Calcular métricas después de calibración
    print("\n>>> Métricas después de calibración:")
    fairness_after = calculate_fairness_metrics(y_true, y_pred_calibrated, groups)
    print(fairness_after.to_string(index=False))
    
    disparity_after = calculate_disparity(fairness_after)
    print(f"\n   Disparidad en predicciones positivas: {disparity_after['disparity']:.4f}")
    print(f"   Umbral cumplido: {disparity_after['threshold_met']}")
    
    # 4. Preparar resultados
    results = {
        'fairness_before': fairness_before,
        'fairness_after': fairness_after,
        'disparity_before': disparity_before,
        'disparity_after': disparity_after,
        'y_pred_calibrated': y_pred_calibrated,
        'thresholds': thresholds,
        'improvement': disparity_before['disparity'] - disparity_after['disparity']
    }
    
    print("=" * 50)
    print(f"✓ Mejora en disparidad: {results['improvement']:.4f}")
    print("=" * 50)
    
    return results
