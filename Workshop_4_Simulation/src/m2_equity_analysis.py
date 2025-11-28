"""
Workshop 4: Módulo M2 - Equity Analysis
Análisis de equidad y detección de sesgo en grupos demográficos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COLUMN, EQUITY_COLUMN, BIAS_THRESHOLD


def analyze_equity_by_group(df, target_col=TARGET_COLUMN, equity_col=EQUITY_COLUMN):
    """
    Calcula estadísticas de equidad por grupo demográfico.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        equity_col: Nombre de la columna de grupo de equidad
    
    Returns:
        DataFrame: Estadísticas por grupo
    """
    if equity_col not in df.columns:
        print(f"⚠ Columna '{equity_col}' no encontrada. Usando análisis general.")
        return pd.DataFrame()
    
    # Calcular estadísticas por grupo
    equity_stats = df.groupby(equity_col).agg({
        target_col: ['count', 'mean', 'sum']
    }).round(4)
    
    equity_stats.columns = ['n_samples', 'event_rate', 'n_events']
    equity_stats = equity_stats.reset_index()
    
    # Calcular porcentaje del total
    equity_stats['pct_total'] = (equity_stats['n_samples'] / equity_stats['n_samples'].sum() * 100).round(2)
    
    print(f"✓ Análisis de equidad por '{equity_col}':")
    print(equity_stats.to_string(index=False))
    
    return equity_stats


def detect_bias(df, target_col=TARGET_COLUMN, equity_col=EQUITY_COLUMN):
    """
    Detecta sesgo calculando la disparidad máxima entre grupos.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        equity_col: Nombre de la columna de grupo de equidad
    
    Returns:
        tuple: (max_disparity, bias_detected, disparity_details)
    """
    if equity_col not in df.columns:
        print(f"⚠ Columna '{equity_col}' no encontrada.")
        return 0.0, False, {}
    
    # Calcular tasa de eventos por grupo
    event_rates = df.groupby(equity_col)[target_col].mean()
    
    # Calcular disparidad máxima
    max_rate = event_rates.max()
    min_rate = event_rates.min()
    max_disparity = max_rate - min_rate
    
    # Determinar si hay sesgo
    bias_detected = max_disparity > BIAS_THRESHOLD
    
    # Detalles de la disparidad
    disparity_details = {
        'max_group': event_rates.idxmax(),
        'max_rate': max_rate,
        'min_group': event_rates.idxmin(),
        'min_rate': min_rate,
        'disparity': max_disparity,
        'threshold': BIAS_THRESHOLD
    }
    
    status = "⚠ SESGO DETECTADO" if bias_detected else "✓ Sin sesgo significativo"
    print(f"\n{status}")
    print(f"  Disparidad máxima: {max_disparity:.4f} (umbral: {BIAS_THRESHOLD})")
    print(f"  Grupo más alto: {disparity_details['max_group']} ({max_rate:.4f})")
    print(f"  Grupo más bajo: {disparity_details['min_group']} ({min_rate:.4f})")
    
    return max_disparity, bias_detected, disparity_details


def plot_equity_analysis(equity_stats, save_path=None):
    """
    Genera visualización del análisis de equidad.
    
    Args:
        equity_stats: DataFrame con estadísticas de equidad
        save_path: Ruta para guardar la figura (opcional)
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if equity_stats.empty:
        print("⚠ No hay datos para visualizar.")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Distribución de muestras por grupo
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(equity_stats))
    bars1 = ax1.bar(equity_stats[EQUITY_COLUMN], equity_stats['n_samples'], color=colors)
    ax1.set_xlabel('Grupo Demográfico')
    ax1.set_ylabel('Número de Muestras')
    ax1.set_title('Distribución de Muestras por Grupo')
    ax1.tick_params(axis='x', rotation=45)
    
    # Añadir etiquetas de porcentaje
    for bar, pct in zip(bars1, equity_stats['pct_total']):
        ax1.annotate(f'{pct:.1f}%', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    # Gráfico 2: Tasa de eventos por grupo
    ax2 = axes[1]
    bars2 = ax2.bar(equity_stats[EQUITY_COLUMN], equity_stats['event_rate'], color=colors)
    ax2.axhline(y=BIAS_THRESHOLD, color='red', linestyle='--', label=f'Umbral: {BIAS_THRESHOLD}')
    ax2.set_xlabel('Grupo Demográfico')
    ax2.set_ylabel('Tasa de Eventos')
    ax2.set_title('Tasa de Eventos por Grupo')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Añadir etiquetas de valor
    for bar in bars2:
        ax2.annotate(f'{bar.get_height():.3f}', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def run_equity_analysis(df):
    """
    Pipeline completo del Módulo M2: Análisis de Equidad.
    
    Args:
        df: DataFrame con los datos preprocesados
    
    Returns:
        dict: Resultados del análisis de equidad
    """
    print("=" * 50)
    print("MÓDULO M2: EQUITY ANALYSIS")
    print("=" * 50)
    
    # 1. Estadísticas por grupo
    equity_stats = analyze_equity_by_group(df)
    
    # 2. Detección de sesgo
    max_disparity, bias_detected, disparity_details = detect_bias(df)
    
    # 3. Preparar resultados
    results = {
        'equity_stats': equity_stats,
        'max_disparity': max_disparity,
        'bias_detected': bias_detected,
        'disparity_details': disparity_details
    }
    
    print("=" * 50)
    print("✓ Análisis de equidad completado")
    print("=" * 50)
    
    return results
