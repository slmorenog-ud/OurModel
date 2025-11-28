"""
Workshop 4: SIMULACIÓN 2 - Event-Based Cellular Automata
Autómatas celulares para modelar comportamiento emergente en supervivencia post-HCT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import pandas as pd
from config import (
    CA_GRID_SIZE, CA_STEPS, CA_RECOVERY_FACTOR, 
    CA_PROGRESSION_FACTOR, CA_CHAOS_FACTOR, TARGET_COLUMN
)


class HCTCellularAutomata:
    """
    Autómata Celular para simular evolución de estados de pacientes post-HCT.
    
    Estados:
        0 = Estable (verde)
        1 = En Riesgo (amarillo)
        2 = Evento (rojo)
    """
    
    # Constantes de estados
    STABLE = 0
    AT_RISK = 1
    EVENT = 2
    
    def __init__(self, grid_size=CA_GRID_SIZE, df_real=None):
        """
        Inicializa el autómata celular.
        
        Args:
            grid_size: Tamaño de la grilla NxN
            df_real: DataFrame con datos reales para inicializar probabilidades
        """
        self.grid_size = grid_size
        self.df_real = df_real
        self.grid = None
        self.history = []
        self.state_counts = []
        
        # Calcular probabilidades iniciales de datos reales
        self._calculate_initial_probs()
        
        # Inicializar grilla
        self._initialize_grid()
    
    def _calculate_initial_probs(self):
        """Calcula probabilidades iniciales basadas en datos reales."""
        if self.df_real is not None and TARGET_COLUMN in self.df_real.columns:
            event_rate = self.df_real[TARGET_COLUMN].mean()
            # Distribuir: stable, at_risk, event
            self.initial_probs = [
                0.6 - event_rate/2,  # Estable
                0.3 + event_rate/4,  # En Riesgo
                0.1 + event_rate/4   # Evento
            ]
        else:
            # Probabilidades por defecto
            self.initial_probs = [0.5, 0.35, 0.15]
        
        # Normalizar
        total = sum(self.initial_probs)
        self.initial_probs = [p/total for p in self.initial_probs]
    
    def _initialize_grid(self):
        """Crea grilla NxN con estados iniciales basados en probabilidades."""
        self.grid = np.random.choice(
            [self.STABLE, self.AT_RISK, self.EVENT],
            size=(self.grid_size, self.grid_size),
            p=self.initial_probs
        )
        self.history = [self.grid.copy()]
        self._update_state_counts()
    
    def _count_neighbors(self, x, y, state):
        """
        Cuenta vecinos en vecindad de Moore (8 vecinos).
        
        Args:
            x, y: Coordenadas de la celda
            state: Estado a contar
        
        Returns:
            int: Número de vecinos con el estado especificado
        """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size
                if self.grid[nx, ny] == state:
                    count += 1
        return count
    
    def _update_state_counts(self):
        """Actualiza conteo de estados."""
        counts = {
            'stable': np.sum(self.grid == self.STABLE),
            'at_risk': np.sum(self.grid == self.AT_RISK),
            'event': np.sum(self.grid == self.EVENT)
        }
        counts['total'] = self.grid_size ** 2
        counts['event_rate'] = counts['event'] / counts['total']
        self.state_counts.append(counts)
    
    def apply_rules(self, recovery_factor=CA_RECOVERY_FACTOR, 
                    progression_factor=CA_PROGRESSION_FACTOR,
                    chaos_factor=CA_CHAOS_FACTOR):
        """
        Aplica reglas de transición del autómata celular.
        
        Reglas:
        - Estable → Riesgo: si ≥3 vecinos en riesgo O evento caótico
        - Riesgo → Evento: si ≥4 vecinos con evento O probabilidad de progresión
        - Riesgo → Estable: probabilidad de recuperación
        - Evento → Riesgo: probabilidad de recuperación parcial
        
        Args:
            recovery_factor: Factor de recuperación
            progression_factor: Factor de progresión
            chaos_factor: Factor de eventos caóticos
        """
        new_grid = self.grid.copy()
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                current_state = self.grid[x, y]
                
                # Contar vecinos
                risk_neighbors = self._count_neighbors(x, y, self.AT_RISK)
                event_neighbors = self._count_neighbors(x, y, self.EVENT)
                
                # Evento caótico aleatorio (Butterfly Effect)
                chaos_event = np.random.random() < chaos_factor
                
                if current_state == self.STABLE:
                    # Estable → Riesgo
                    if risk_neighbors >= 3 or event_neighbors >= 2 or chaos_event:
                        new_grid[x, y] = self.AT_RISK
                
                elif current_state == self.AT_RISK:
                    # Riesgo → Evento
                    if event_neighbors >= 4 or np.random.random() < progression_factor:
                        new_grid[x, y] = self.EVENT
                    # Riesgo → Estable (recuperación)
                    elif np.random.random() < recovery_factor:
                        new_grid[x, y] = self.STABLE
                
                elif current_state == self.EVENT:
                    # Evento → Riesgo (recuperación parcial)
                    if np.random.random() < recovery_factor * 0.5:
                        new_grid[x, y] = self.AT_RISK
        
        self.grid = new_grid
        self.history.append(self.grid.copy())
        self._update_state_counts()
    
    def run_simulation(self, steps=CA_STEPS, recovery_factor=CA_RECOVERY_FACTOR,
                       progression_factor=CA_PROGRESSION_FACTOR,
                       chaos_factor=CA_CHAOS_FACTOR):
        """
        Ejecuta la simulación por un número de pasos.
        
        Args:
            steps: Número de pasos de simulación
            recovery_factor: Factor de recuperación
            progression_factor: Factor de progresión
            chaos_factor: Factor de eventos caóticos
        
        Returns:
            DataFrame: Evolución de estados por paso
        """
        print(f"\n>>> Ejecutando simulación de autómatas celulares...")
        print(f"   Tamaño de grilla: {self.grid_size}x{self.grid_size}")
        print(f"   Pasos: {steps}")
        print(f"   Factores: recuperación={recovery_factor}, progresión={progression_factor}, caos={chaos_factor}")
        
        for step in range(steps):
            self.apply_rules(recovery_factor, progression_factor, chaos_factor)
            
            if (step + 1) % 20 == 0:
                counts = self.state_counts[-1]
                print(f"   Paso {step+1}: Estables={counts['stable']}, "
                      f"En Riesgo={counts['at_risk']}, Eventos={counts['event']} "
                      f"(Tasa: {counts['event_rate']:.3f})")
        
        return pd.DataFrame(self.state_counts)
    
    def get_emergence_metrics(self):
        """
        Calcula métricas de comportamiento emergente.
        
        Returns:
            dict: Métricas de emergencia
        """
        df_states = pd.DataFrame(self.state_counts)
        
        metrics = {
            'initial_event_rate': df_states['event_rate'].iloc[0],
            'final_event_rate': df_states['event_rate'].iloc[-1],
            'max_event_rate': df_states['event_rate'].max(),
            'min_event_rate': df_states['event_rate'].min(),
            'event_rate_std': df_states['event_rate'].std(),
            'trend': 'increasing' if df_states['event_rate'].iloc[-1] > df_states['event_rate'].iloc[0] else 'decreasing',
            'volatility': df_states['event_rate'].diff().abs().mean()
        }
        
        return metrics


def plot_automata_evolution(ca, save_path=None):
    """
    Genera visualización de la evolución del autómata.
    
    Args:
        ca: Instancia de HCTCellularAutomata
        save_path: Ruta para guardar la figura
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Colormap para estados
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Verde, Amarillo, Rojo
    cmap = ListedColormap(colors)
    
    # Seleccionar pasos representativos
    n_history = len(ca.history)
    steps_to_show = [0, n_history//4, n_history//2, 3*n_history//4, n_history-1]
    
    # Gráficos de grilla en diferentes pasos (5 slots: positions 0-4 in 2x3 grid)
    # Layout: [0,0], [0,1], [0,2], [1,0], [1,1] for steps, [1,2] for evolution
    plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for idx, step in enumerate(steps_to_show[:5]):
        row, col = plot_positions[idx]
        ax = axes[row, col]
        im = ax.imshow(ca.history[step], cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f'Paso {step}')
        ax.axis('off')
    
    # Gráfico de evolución temporal
    ax_evolution = axes[1, 2]
    df_states = pd.DataFrame(ca.state_counts)
    
    ax_evolution.plot(df_states.index, df_states['stable'] / ca.grid_size**2 * 100, 
                      'g-', label='Estable', linewidth=2)
    ax_evolution.plot(df_states.index, df_states['at_risk'] / ca.grid_size**2 * 100, 
                      'y-', label='En Riesgo', linewidth=2)
    ax_evolution.plot(df_states.index, df_states['event'] / ca.grid_size**2 * 100, 
                      'r-', label='Evento', linewidth=2)
    ax_evolution.set_xlabel('Paso')
    ax_evolution.set_ylabel('Porcentaje de Celdas (%)')
    ax_evolution.set_title('Evolución de Estados')
    ax_evolution.legend()
    ax_evolution.grid(True, alpha=0.3)
    
    # Leyenda de colores
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='#2ecc71', label='Estable'),
        plt.Rectangle((0,0), 1, 1, facecolor='#f1c40f', label='En Riesgo'),
        plt.Rectangle((0,0), 1, 1, facecolor='#e74c3c', label='Evento')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return fig


def compare_scenarios(df_real=None, save_path=None):
    """
    Compara diferentes escenarios de simulación.
    
    Args:
        df_real: DataFrame con datos reales (opcional)
        save_path: Ruta para guardar la figura
    
    Returns:
        dict: Resultados de comparación
    """
    scenarios = {
        'Baseline': {'recovery': 0.08, 'progression': 0.12, 'chaos': 0.03},
        'Alta Recuperación': {'recovery': 0.15, 'progression': 0.12, 'chaos': 0.03},
        'Alta Progresión': {'recovery': 0.08, 'progression': 0.20, 'chaos': 0.03},
        'Alto Caos': {'recovery': 0.08, 'progression': 0.12, 'chaos': 0.10}
    }
    
    results = {}
    
    print("\n>>> Comparando escenarios de simulación...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, params in scenarios.items():
        ca = HCTCellularAutomata(grid_size=30, df_real=df_real)
        ca.run_simulation(
            steps=50,
            recovery_factor=params['recovery'],
            progression_factor=params['progression'],
            chaos_factor=params['chaos']
        )
        
        df_states = pd.DataFrame(ca.state_counts)
        ax.plot(df_states.index, df_states['event_rate'] * 100, 
                label=name, linewidth=2)
        
        metrics = ca.get_emergence_metrics()
        results[name] = {
            'params': params,
            'metrics': metrics,
            'final_event_rate': metrics['final_event_rate']
        }
        
        print(f"   {name}: Tasa final de eventos = {metrics['final_event_rate']:.3f}")
    
    ax.set_xlabel('Paso de Simulación')
    ax.set_ylabel('Tasa de Eventos (%)')
    ax.set_title('Comparación de Escenarios - Autómatas Celulares')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    return results


def run_simulation2(df=None):
    """
    Pipeline completo de la Simulación 2: Autómatas Celulares.
    
    Args:
        df: DataFrame con los datos preprocesados (opcional)
    
    Returns:
        dict: Resultados completos de la simulación
    """
    print("=" * 60)
    print("SIMULACIÓN 2: EVENT-BASED CELLULAR AUTOMATA")
    print("=" * 60)
    
    # 1. Crear autómata celular
    print("\n>>> Inicializando autómata celular...")
    ca = HCTCellularAutomata(grid_size=CA_GRID_SIZE, df_real=df)
    
    initial_counts = ca.state_counts[-1]
    print(f"   Estado inicial:")
    print(f"   - Estables: {initial_counts['stable']} ({initial_counts['stable']/initial_counts['total']*100:.1f}%)")
    print(f"   - En Riesgo: {initial_counts['at_risk']} ({initial_counts['at_risk']/initial_counts['total']*100:.1f}%)")
    print(f"   - Eventos: {initial_counts['event']} ({initial_counts['event']/initial_counts['total']*100:.1f}%)")
    
    # 2. Ejecutar simulación principal
    df_evolution = ca.run_simulation(steps=CA_STEPS)
    
    # 3. Calcular métricas de emergencia
    emergence_metrics = ca.get_emergence_metrics()
    
    print("\n>>> Métricas de Comportamiento Emergente:")
    print(f"   Tasa inicial de eventos: {emergence_metrics['initial_event_rate']:.4f}")
    print(f"   Tasa final de eventos: {emergence_metrics['final_event_rate']:.4f}")
    print(f"   Tasa máxima de eventos: {emergence_metrics['max_event_rate']:.4f}")
    print(f"   Volatilidad: {emergence_metrics['volatility']:.4f}")
    print(f"   Tendencia: {emergence_metrics['trend']}")
    
    # 4. Comparar escenarios
    scenario_results = compare_scenarios(df_real=df)
    
    # 5. Preparar resultados
    simulation_results = {
        'automata': ca,
        'evolution_df': df_evolution,
        'emergence_metrics': emergence_metrics,
        'scenario_comparison': scenario_results,
        'final_grid': ca.grid.copy(),
        'history': ca.history
    }
    
    print("\n" + "=" * 60)
    print("RESUMEN SIMULACIÓN 2")
    print("=" * 60)
    print(f"✓ Pasos ejecutados: {CA_STEPS}")
    print(f"✓ Tamaño de grilla: {CA_GRID_SIZE}x{CA_GRID_SIZE}")
    print(f"✓ Tasa final de eventos: {emergence_metrics['final_event_rate']:.4f}")
    print(f"✓ Tendencia observada: {emergence_metrics['trend']}")
    print(f"✓ Escenarios comparados: {len(scenario_results)}")
    print("=" * 60)
    
    return simulation_results
