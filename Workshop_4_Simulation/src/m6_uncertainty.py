"""
Workshop 4: Módulo M6 - Uncertainty Quantification
Cuantificación de incertidumbre en predicciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from config import RANDOM_STATE, N_BOOTSTRAP, INSTABILITY_THRESHOLD


def bootstrap_predictions(model, X, y, n_bootstrap=N_BOOTSTRAP):
    """
    Genera predicciones con bootstrap para estimar incertidumbre.
    
    Args:
        model: Modelo a usar
        X: Features
        y: Target
        n_bootstrap: Número de muestras bootstrap
    
    Returns:
        tuple: (predictions_matrix, mean_predictions, std_predictions)
    """
    n_samples = len(X)
    predictions_matrix = np.zeros((n_bootstrap, n_samples))
    
    print(f">>> Ejecutando {n_bootstrap} iteraciones de bootstrap...")
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        # Entrenar modelo
        model_boot = model.__class__(**model.get_params())
        model_boot.fit(X_boot, y_boot)
        
        # Predecir probabilidades
        predictions_matrix[i] = model_boot.predict_proba(X)[:, 1]
        
        if (i + 1) % 5 == 0:
            print(f"   Completado: {i+1}/{n_bootstrap}")
    
    mean_predictions = predictions_matrix.mean(axis=0)
    std_predictions = predictions_matrix.std(axis=0)
    
    return predictions_matrix, mean_predictions, std_predictions


def calculate_prediction_intervals(predictions_matrix, confidence=0.95):
    """
    Calcula intervalos de confianza para las predicciones.
    
    Args:
        predictions_matrix: Matriz de predicciones bootstrap
        confidence: Nivel de confianza
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(predictions_matrix, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions_matrix, upper_percentile, axis=0)
    
    return lower_bound, upper_bound


def calculate_uncertainty_metrics(std_predictions):
    """
    Calcula métricas de incertidumbre.
    
    Args:
        std_predictions: Desviación estándar de predicciones
    
    Returns:
        dict: Métricas de incertidumbre
    """
    metrics = {
        'mean_uncertainty': np.mean(std_predictions),
        'max_uncertainty': np.max(std_predictions),
        'min_uncertainty': np.min(std_predictions),
        'median_uncertainty': np.median(std_predictions),
        'high_uncertainty_pct': np.mean(std_predictions > INSTABILITY_THRESHOLD) * 100
    }
    
    return metrics


def identify_uncertain_predictions(mean_predictions, std_predictions, threshold=None):
    """
    Identifica predicciones con alta incertidumbre.
    
    Args:
        mean_predictions: Media de predicciones
        std_predictions: Desviación estándar de predicciones
        threshold: Umbral de incertidumbre
    
    Returns:
        DataFrame: Predicciones ordenadas por incertidumbre
    """
    if threshold is None:
        threshold = INSTABILITY_THRESHOLD
    
    df_uncertainty = pd.DataFrame({
        'mean_prediction': mean_predictions,
        'std_prediction': std_predictions,
        'high_uncertainty': std_predictions > threshold,
        'confidence': 1 - std_predictions
    })
    
    df_uncertainty = df_uncertainty.sort_values('std_prediction', ascending=False)
    
    n_high = df_uncertainty['high_uncertainty'].sum()
    print(f"✓ Predicciones con alta incertidumbre: {n_high} ({n_high/len(df_uncertainty)*100:.1f}%)")
    
    return df_uncertainty


def plot_uncertainty_analysis(mean_predictions, std_predictions, lower_bound, upper_bound, save_path=None):
    """
    Genera visualización del análisis de incertidumbre.
    
    Args:
        mean_predictions: Media de predicciones
        std_predictions: Desviación estándar de predicciones
        lower_bound: Límite inferior del intervalo
        upper_bound: Límite superior del intervalo
        save_path: Ruta para guardar la figura
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Distribución de incertidumbre
    ax1 = axes[0, 0]
    ax1.hist(std_predictions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=INSTABILITY_THRESHOLD, color='red', linestyle='--', 
                label=f'Umbral: {INSTABILITY_THRESHOLD}')
    ax1.axvline(x=np.mean(std_predictions), color='green', linestyle='-',
                label=f'Media: {np.mean(std_predictions):.3f}')
    ax1.set_xlabel('Incertidumbre (Desviación Estándar)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Incertidumbre en Predicciones')
    ax1.legend()
    
    # Gráfico 2: Predicción media vs Incertidumbre
    ax2 = axes[0, 1]
    scatter = ax2.scatter(mean_predictions, std_predictions, 
                          c=mean_predictions, cmap='RdYlGn_r', alpha=0.5, s=10)
    ax2.axhline(y=INSTABILITY_THRESHOLD, color='red', linestyle='--')
    ax2.set_xlabel('Predicción Media')
    ax2.set_ylabel('Incertidumbre')
    ax2.set_title('Predicción vs Incertidumbre')
    plt.colorbar(scatter, ax=ax2, label='Predicción')
    
    # Gráfico 3: Intervalos de confianza (muestra de 50 predicciones)
    ax3 = axes[1, 0]
    n_sample = min(50, len(mean_predictions))
    indices = np.random.choice(len(mean_predictions), n_sample, replace=False)
    indices = np.sort(indices)
    
    x_positions = range(n_sample)
    ax3.errorbar(x_positions, mean_predictions[indices], 
                 yerr=[mean_predictions[indices] - lower_bound[indices],
                       upper_bound[indices] - mean_predictions[indices]],
                 fmt='o', color='steelblue', ecolor='lightgray', 
                 elinewidth=2, capsize=3, markersize=4)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Umbral 0.5')
    ax3.set_xlabel('Muestra')
    ax3.set_ylabel('Probabilidad Predicha')
    ax3.set_title(f'Intervalos de Confianza 95% (muestra de {n_sample})')
    ax3.legend()
    
    # Gráfico 4: Boxplot por rango de predicción
    ax4 = axes[1, 1]
    
    # Crear bins de predicción
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0']
    prediction_bins = pd.cut(mean_predictions, bins=bins, labels=labels)
    
    df_boxplot = pd.DataFrame({
        'prediction_range': prediction_bins,
        'uncertainty': std_predictions
    })
    
    df_boxplot.boxplot(column='uncertainty', by='prediction_range', ax=ax4)
    ax4.axhline(y=INSTABILITY_THRESHOLD, color='red', linestyle='--')
    ax4.set_xlabel('Rango de Predicción')
    ax4.set_ylabel('Incertidumbre')
    ax4.set_title('Incertidumbre por Rango de Predicción')
    plt.suptitle('')  # Remover título automático de boxplot
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def run_uncertainty_quantification(model, X, y):
    """
    Pipeline completo del Módulo M6: Uncertainty Quantification.
    
    Args:
        model: Modelo entrenado
        X: Features
        y: Target
    
    Returns:
        dict: Resultados de cuantificación de incertidumbre
    """
    print("=" * 50)
    print("MÓDULO M6: UNCERTAINTY QUANTIFICATION")
    print("=" * 50)
    
    # 1. Bootstrap predictions
    predictions_matrix, mean_predictions, std_predictions = bootstrap_predictions(
        model, X, y, N_BOOTSTRAP
    )
    
    # 2. Calcular intervalos de confianza
    lower_bound, upper_bound = calculate_prediction_intervals(predictions_matrix)
    
    # 3. Calcular métricas
    uncertainty_metrics = calculate_uncertainty_metrics(std_predictions)
    
    print("\n>>> Métricas de Incertidumbre:")
    print(f"   Media: {uncertainty_metrics['mean_uncertainty']:.4f}")
    print(f"   Máxima: {uncertainty_metrics['max_uncertainty']:.4f}")
    print(f"   Mínima: {uncertainty_metrics['min_uncertainty']:.4f}")
    print(f"   Predicciones con alta incertidumbre: {uncertainty_metrics['high_uncertainty_pct']:.1f}%")
    
    # 4. Identificar predicciones inciertas
    df_uncertainty = identify_uncertain_predictions(mean_predictions, std_predictions)
    
    # 5. Preparar resultados
    results = {
        'predictions_matrix': predictions_matrix,
        'mean_predictions': mean_predictions,
        'std_predictions': std_predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'uncertainty_metrics': uncertainty_metrics,
        'df_uncertainty': df_uncertainty
    }
    
    # Verificar umbral de estabilidad
    stability_ok = uncertainty_metrics['mean_uncertainty'] <= INSTABILITY_THRESHOLD
    
    print("=" * 50)
    if stability_ok:
        print(f"✓ Modelo ESTABLE (incertidumbre media ≤ {INSTABILITY_THRESHOLD})")
    else:
        print(f"⚠ Modelo INESTABLE (incertidumbre media > {INSTABILITY_THRESHOLD})")
    print("=" * 50)
    
    results['stability_ok'] = stability_ok
    
    return results
