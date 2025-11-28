"""
Workshop 4: Configuración y Constantes del Sistema
Basado en Workshop 2 (System Design) y Workshop 3 (Quality Control)
"""

# COLUMNAS DEL DATASET
TARGET_COLUMN = 'efs'
TIME_COLUMN = 'efs_time'
EQUITY_COLUMN = 'race_group'
ID_COLUMN = 'ID'

# UMBRALES DE CALIDAD (Workshop 3 - Table 3)
MISSING_THRESHOLD = 0.30
INSTABILITY_THRESHOLD = 0.15
BIAS_THRESHOLD = 0.10
ACCURACY_TARGET = 0.70

# PARÁMETROS DE SIMULACIÓN
RANDOM_STATE = 42
SAMPLE_SIZE = 0.5
N_ITERATIONS = 5
N_BOOTSTRAP = 10

# PARÁMETROS DE AUTÓMATAS CELULARES
CA_GRID_SIZE = 40
CA_STEPS = 80
CA_RECOVERY_FACTOR = 0.08
CA_PROGRESSION_FACTOR = 0.12
CA_CHAOS_FACTOR = 0.03

# VARIABLES CRÍTICAS (Workshop 1 - Sensitivity Analysis)
CRITICAL_FEATURES = [
    'age_at_hct', 'dri_score', 'comorbidity_score', 'karnofsky_score',
    'donor_age', 'hla_high_res_10', 'hla_high_res_8', 'year_hct',
    'conditioning_intensity', 'graft_type', 'prim_disease_hct', 'cmv_status'
]
