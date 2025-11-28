"""
Workshop 4: SIMULACIÓN 1 - Data-Driven Machine Learning
Modelo de ML para predecir supervivencia post-HCT con análisis de variabilidad y caos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from config import (
    TARGET_COLUMN, RANDOM_STATE, N_ITERATIONS, 
    INSTABILITY_THRESHOLD, ACCURACY_TARGET, ID_COLUMN
)


def prepare_simulation_data(df, features):
    """
    Prepara los datos para la simulación.
    
    Args:
        df: DataFrame con los datos
        features: Lista de features a usar
    
    Returns:
        tuple: (X, y) preparados
    """
    # Filtrar features presentes en el DataFrame
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Remover filas con target faltante
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"✓ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} features")
    
    return X, y


def run_ml_iterations(X, y, n_iterations=N_ITERATIONS):
    """
    Ejecuta el modelo ML con diferentes semillas para observar variabilidad.
    
    Args:
        X: Features
        y: Target
        n_iterations: Número de iteraciones
    
    Returns:
        dict: Resultados de todas las iteraciones
    """
    results = {
        'accuracy': [],
        'auc': [],
        'models': [],
        'seeds': []
    }
    
    print(f"\n>>> Ejecutando {n_iterations} iteraciones del modelo...")
    
    for i in range(n_iterations):
        seed = RANDOM_STATE + i * 10
        results['seeds'].append(seed)
        
        # Split datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Entrenar modelo
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed
        )
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results['accuracy'].append(acc)
        results['auc'].append(auc)
        results['models'].append(model)
        
        print(f"   Iteración {i+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")
    
    return results


def analyze_variability(results):
    """
    Analiza la variabilidad del modelo y verifica contra umbrales.
    
    Args:
        results: Resultados de las iteraciones
    
    Returns:
        dict: Análisis de variabilidad
    """
    accuracy_array = np.array(results['accuracy'])
    auc_array = np.array(results['auc'])
    
    analysis = {
        'accuracy_mean': np.mean(accuracy_array),
        'accuracy_std': np.std(accuracy_array),
        'auc_mean': np.mean(auc_array),
        'auc_std': np.std(auc_array),
        'accuracy_cv': np.std(accuracy_array) / np.mean(accuracy_array),
        'auc_cv': np.std(auc_array) / np.mean(auc_array),
    }
    
    # Verificar umbrales
    analysis['stability_ok'] = analysis['accuracy_cv'] <= INSTABILITY_THRESHOLD
    analysis['accuracy_ok'] = analysis['accuracy_mean'] >= ACCURACY_TARGET
    
    print("\n>>> Análisis de Variabilidad:")
    print(f"   Accuracy: {analysis['accuracy_mean']:.4f} ± {analysis['accuracy_std']:.4f}")
    print(f"   AUC:      {analysis['auc_mean']:.4f} ± {analysis['auc_std']:.4f}")
    print(f"   CV Accuracy: {analysis['accuracy_cv']:.4f} (umbral: {INSTABILITY_THRESHOLD})")
    
    if analysis['stability_ok']:
        print(f"   ✓ Modelo ESTABLE (CV ≤ {INSTABILITY_THRESHOLD})")
    else:
        print(f"   ⚠ Modelo INESTABLE (CV > {INSTABILITY_THRESHOLD})")
    
    if analysis['accuracy_ok']:
        print(f"   ✓ Accuracy cumple objetivo (≥ {ACCURACY_TARGET})")
    else:
        print(f"   ⚠ Accuracy por debajo del objetivo (< {ACCURACY_TARGET})")
    
    return analysis


def chaos_sensitivity_analysis(X, y, perturbation_levels=None):
    """
    Análisis de sensibilidad al caos (Butterfly Effect).
    Añade ruido gaussiano y mide el impacto en las predicciones.
    
    Args:
        X: Features
        y: Target
        perturbation_levels: Niveles de perturbación a probar
    
    Returns:
        DataFrame: Resultados del análisis de caos
    """
    if perturbation_levels is None:
        perturbation_levels = [0.01, 0.05, 0.10, 0.15]
    
    print("\n>>> Análisis de Sensibilidad al Caos (Butterfly Effect):")
    
    # Entrenar modelo base
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    base_model.fit(X_train, y_train)
    
    # Predicción base
    base_pred = base_model.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    base_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])
    
    chaos_results = []
    
    for level in perturbation_levels:
        # Añadir ruido gaussiano
        noise = np.random.normal(0, level, X_test.shape)
        X_test_noisy = X_test + noise
        
        # Predicción con ruido
        noisy_pred = base_model.predict(X_test_noisy)
        noisy_proba = base_model.predict_proba(X_test_noisy)[:, 1]
        
        noisy_acc = accuracy_score(y_test, noisy_pred)
        noisy_auc = roc_auc_score(y_test, noisy_proba)
        
        # Calcular cambio en predicciones
        pred_change = np.mean(base_pred != noisy_pred)
        
        chaos_results.append({
            'perturbation': level,
            'accuracy': noisy_acc,
            'auc': noisy_auc,
            'accuracy_drop': base_acc - noisy_acc,
            'auc_drop': base_auc - noisy_auc,
            'prediction_change': pred_change
        })
        
        print(f"   Perturbación {level*100:.0f}%: Acc={noisy_acc:.4f} (Δ={base_acc-noisy_acc:+.4f}), "
              f"Cambio predicciones={pred_change*100:.1f}%")
    
    return pd.DataFrame(chaos_results)


def plot_simulation1_results(results_df, chaos_df, best_model, features, save_path=None):
    """
    Genera visualizaciones de la Simulación 1.
    
    Args:
        results_df: DataFrame con resultados de iteraciones
        chaos_df: DataFrame con resultados de análisis de caos
        best_model: Mejor modelo entrenado
        features: Lista de features
        save_path: Ruta para guardar la figura
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Variabilidad de Accuracy
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(results_df['accuracy'])+1), results_df['accuracy'], 
             'o-', color='blue', markersize=8, linewidth=2)
    ax1.axhline(y=np.mean(results_df['accuracy']), color='red', linestyle='--', 
                label=f'Media: {np.mean(results_df["accuracy"]):.4f}')
    ax1.axhline(y=ACCURACY_TARGET, color='green', linestyle=':', 
                label=f'Objetivo: {ACCURACY_TARGET}')
    ax1.fill_between(range(1, len(results_df['accuracy'])+1),
                     np.mean(results_df['accuracy']) - np.std(results_df['accuracy']),
                     np.mean(results_df['accuracy']) + np.std(results_df['accuracy']),
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Variabilidad de Accuracy entre Iteraciones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Variabilidad de AUC
    ax2 = axes[0, 1]
    ax2.plot(range(1, len(results_df['auc'])+1), results_df['auc'], 
             'o-', color='purple', markersize=8, linewidth=2)
    ax2.axhline(y=np.mean(results_df['auc']), color='red', linestyle='--',
                label=f'Media: {np.mean(results_df["auc"]):.4f}')
    ax2.fill_between(range(1, len(results_df['auc'])+1),
                     np.mean(results_df['auc']) - np.std(results_df['auc']),
                     np.mean(results_df['auc']) + np.std(results_df['auc']),
                     alpha=0.2, color='purple')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('AUC')
    ax2.set_title('Variabilidad de AUC entre Iteraciones')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Análisis de Caos
    ax3 = axes[1, 0]
    ax3.plot(chaos_df['perturbation'] * 100, chaos_df['accuracy'], 
             'o-', color='red', markersize=8, linewidth=2, label='Accuracy')
    ax3.plot(chaos_df['perturbation'] * 100, chaos_df['auc'], 
             's-', color='orange', markersize=8, linewidth=2, label='AUC')
    ax3.set_xlabel('Nivel de Perturbación (%)')
    ax3.set_ylabel('Métrica')
    ax3.set_title('Análisis de Sensibilidad al Caos (Butterfly Effect)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Importancia de Features (del mejor modelo)
    ax4 = axes[1, 1]
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(10)
        
        colors = sns.color_palette("viridis", len(importance_df))
        ax4.barh(importance_df['feature'], importance_df['importance'], color=colors)
        ax4.set_xlabel('Importancia')
        ax4.set_title('Top 10 Features - Mejor Modelo')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def run_simulation1(df, features):
    """
    Pipeline completo de la Simulación 1: Data-Driven ML.
    
    Args:
        df: DataFrame con los datos preprocesados
        features: Lista de features seleccionadas
    
    Returns:
        dict: Resultados completos de la simulación
    """
    print("=" * 60)
    print("SIMULACIÓN 1: DATA-DRIVEN MACHINE LEARNING")
    print("=" * 60)
    
    # 1. Preparar datos
    X, y = prepare_simulation_data(df, features)
    
    # 2. Ejecutar iteraciones del modelo
    iteration_results = run_ml_iterations(X, y, N_ITERATIONS)
    
    # 3. Analizar variabilidad
    variability_analysis = analyze_variability(iteration_results)
    
    # 4. Análisis de sensibilidad al caos
    chaos_results = chaos_sensitivity_analysis(X, y)
    
    # 5. Seleccionar mejor modelo
    best_idx = np.argmax(iteration_results['auc'])
    best_model = iteration_results['models'][best_idx]
    
    # 6. Preparar resultados
    results_df = pd.DataFrame({
        'iteration': range(1, N_ITERATIONS + 1),
        'seed': iteration_results['seeds'],
        'accuracy': iteration_results['accuracy'],
        'auc': iteration_results['auc']
    })
    
    simulation_results = {
        'results_df': results_df,
        'chaos_df': chaos_results,
        'variability_analysis': variability_analysis,
        'best_model': best_model,
        'best_accuracy': iteration_results['accuracy'][best_idx],
        'best_auc': iteration_results['auc'][best_idx],
        'features': list(X.columns)
    }
    
    print("\n" + "=" * 60)
    print("RESUMEN SIMULACIÓN 1")
    print("=" * 60)
    print(f"✓ Iteraciones ejecutadas: {N_ITERATIONS}")
    print(f"✓ Mejor Accuracy: {simulation_results['best_accuracy']:.4f}")
    print(f"✓ Mejor AUC: {simulation_results['best_auc']:.4f}")
    print(f"✓ Estabilidad: {'CUMPLE' if variability_analysis['stability_ok'] else 'NO CUMPLE'}")
    print(f"✓ Objetivo Accuracy: {'CUMPLE' if variability_analysis['accuracy_ok'] else 'NO CUMPLE'}")
    print("=" * 60)
    
    return simulation_results
