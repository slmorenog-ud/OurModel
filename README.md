# Workshop 4: Kaggle System Simulation
## CIBMTR - Equity in post-HCT Survival Predictions

### 📋 Descripción
Este proyecto implementa las simulaciones requeridas para el Workshop 4 del curso de Análisis y Diseño de Sistemas.  Contiene dos escenarios de simulación para validar la arquitectura del sistema diseñado en workshops anteriores.

### 🎯 Simulaciones Implementadas

| Simulación | Tipo | Descripción |
|------------|------|-------------|
| **Escenario 1** | Data-Driven (ML) | Modelo de ML clásico para predecir supervivencia post-HCT |
| **Escenario 2** | Event-Based (CA) | Autómatas celulares para modelar comportamiento emergente |

### 📁 Estructura del Proyecto

```
Workshop_4_Simulation/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv                 # Dataset de Kaggle (subir manualmente)
│   └── data_dictionary.csv       # Diccionario de datos
├── notebooks/
│   └── Workshop_4_Complete.ipynb # Notebook para Google Colab
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuración y constantes
│   ├── m1_preprocessing.py       # Módulo M1: Preprocesamiento
│   ├── m2_equity_analysis.py     # Módulo M2: Análisis de equidad
│   ├── m3_feature_selection.py   # Módulo M3: Selección de features
│   ├── simulation1_ml.py         # SIMULACIÓN 1: Machine Learning
│   ├── simulation2_automata.py   # SIMULACIÓN 2: Autómatas Celulares
│   ├── m5_fairness. py            # Módulo M5: Calibración de equidad
│   └── m6_uncertainty.py         # Módulo M6: Incertidumbre
├── results/                      # Gráficos generados
└── docs/                         # Informe PDF final
```

### 👥 División del Trabajo (4 personas)

| Persona | Responsabilidad | Archivos |
|---------|-----------------|----------|
| **1** | Datos y Documentación | `m1_preprocessing.py`, `m2_equity_analysis.py`, README |
| **2** | Simulación 1 (ML) | `m3_feature_selection.py`, `simulation1_ml.py` |
| **3** | Simulación 2 (CA) | `simulation2_automata.py` |
| **4** | Validación y Reporte | `m5_fairness.py`, `m6_uncertainty.py`, Informe PDF |

### 🚀 Cómo Ejecutar

#### Opción 1: Google Colab (Recomendado)
1. Abrir `notebooks/Workshop_4_Complete.ipynb` en Google Colab
2. Subir `train.csv` y `data_dictionary.csv` cuando se solicite
3. Ejecutar todas las celdas

#### Opción 2: Local
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook Workshop_4_Complete.ipynb
```

### 📊 Métricas de Éxito (Workshop 3)

| Métrica | Umbral | Descripción |
|---------|--------|-------------|
| `ACCURACY_TARGET` | ≥ 0.70 | C-index mínimo |
| `BIAS_THRESHOLD` | ≤ 0.10 | Disparidad máxima entre grupos |
| `INSTABILITY_THRESHOLD` | ≤ 0.15 | Variabilidad máxima del modelo |

### 📚 Referencias
- Workshop 1: Análisis de Sistemas
- Workshop 2: Diseño del Sistema (Arquitectura M1-M7)
- Workshop 3: Gestión de Proyecto y Control de Calidad
- [Kaggle Competition](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)

### 📅 Fecha de Entrega
Sábado, 29 de Noviembre de 2025, 8:00 AM