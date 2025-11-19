"""
Validação temporal com embargo para séries temporais financeiras.
Implementa TimeSeriesSplit com gap (embargo) entre treino e teste.

Alinhado ao Plano de Pesquisa TCC USP:
- n_splits = 5 (fixo)
- embargo = 1 dia (evita look-ahead bias)
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple


class TimeSeriesSplitWithEmbargo(TimeSeriesSplit):
    """
    TimeSeriesSplit com embargo (gap) entre treino e teste.
    
    O embargo remove os últimos N dias do conjunto de treino antes de cada fold,
    garantindo que não haja informação "vazando" do período de teste para o treino.
    
    Isso é crucial em séries temporais financeiras, onde:
    - Notícias de D0 podem influenciar preços em D+1
    - Precisamos simular condições realistas de trading
    
    Conforme especificado no Plano de Pesquisa:
    - n_splits = 5 (validação walk-forward com 5 folds)
    - embargo = 1 dia (mínimo para evitar look-ahead)
    
    Example:
        >>> from src.utils.validation import TimeSeriesSplitWithEmbargo
        >>> from src.config.constants import N_SPLITS_TIMESERIES, EMBARGO_DAYS
        >>> 
        >>> tscv = TimeSeriesSplitWithEmbargo(
        ...     n_splits=N_SPLITS_TIMESERIES,
        ...     embargo=EMBARGO_DAYS
        ... )
        >>> 
        >>> for train_idx, test_idx in tscv.split(X):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     y_train, y_test = y[train_idx], y[test_idx]
        ...     # Treinar e avaliar modelo
    """
    
    def __init__(self, n_splits: int = 5, embargo: int = 1, 
                 max_train_size: int = None, test_size: int = None):
        """
        Inicializa TimeSeriesSplit com embargo.
        
        Args:
            n_splits: Número de folds (padrão: 5, conforme plano de pesquisa)
            embargo: Número de observações a remover do fim do treino (padrão: 1 dia)
            max_train_size: Tamanho máximo da janela de treino (None = expansível)
            test_size: Tamanho fixo do conjunto de teste (None = variável)
        """
        super().__init__(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size
        )
        self.embargo = embargo
        
        if embargo < 0:
            raise ValueError("embargo deve ser >= 0")
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera índices de treino/teste com embargo aplicado.
        
        Args:
            X: Array-like de features
            y: Array-like de targets (opcional)
            groups: Grupos (ignorado, mantido para compatibilidade)
        
        Yields:
            Tuplas (train_indices, test_indices) para cada fold
        """
        # Obter splits do TimeSeriesSplit padrão
        for train_idx, test_idx in super().split(X, y, groups):
            
            # Aplicar embargo: remover últimos 'embargo' dias do treino
            if self.embargo > 0 and len(train_idx) > self.embargo:
                train_idx = train_idx[:-self.embargo]
            
            # Apenas retornar se ainda houver dados de treino suficientes
            if len(train_idx) >= 2:  # Mínimo de 2 observações para treinar
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna número de folds."""
        return self.n_splits
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"n_splits={self.n_splits}, "
                f"embargo={self.embargo}, "
                f"max_train_size={self.max_train_size}, "
                f"test_size={self.test_size})")


def validate_model_timeseries(model, X, y, n_splits: int = 5, embargo: int = 1, 
                              metrics: dict = None, verbose: bool = True):
    """
    Valida modelo usando walk-forward com embargo.
    
    Função utilitária para simplificar validação de modelos em notebooks.
    
    Args:
        model: Modelo scikit-learn (com fit/predict)
        X: Features (matriz numpy ou sparse)
        y: Target (array numpy)
        n_splits: Número de folds (padrão: 5)
        embargo: Dias de embargo (padrão: 1)
        metrics: Dict de funções de métrica {nome: função}
                 Padrão: AUC e MDA
        verbose: Se True, imprime progresso
    
    Returns:
        Dict com resultados: {
            'scores': {metric_name: [score_fold1, score_fold2, ...]},
            'mean': {metric_name: mean_score},
            'std': {metric_name: std_score}
        }
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.metrics import roc_auc_score, accuracy_score
        >>> 
        >>> model = LogisticRegression(max_iter=2000)
        >>> results = validate_model_timeseries(
        ...     model, X, y,
        ...     metrics={'AUC': roc_auc_score, 'Accuracy': accuracy_score}
        ... )
        >>> print(f"AUC médio: {results['mean']['AUC']:.3f} ± {results['std']['AUC']:.3f}")
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.base import clone
    
    # Métricas padrão se não especificadas
    if metrics is None:
        def mda(y_true, y_pred):
            """Mean Directional Accuracy"""
            return (y_true == (y_pred > 0.5).astype(int)).mean()
        
        metrics = {
            'AUC': roc_auc_score,
            'MDA': mda
        }
    
    # Inicializar armazenamento de scores
    scores = {name: [] for name in metrics.keys()}
    
    # TimeSeriesSplit com embargo
    tscv = TimeSeriesSplitWithEmbargo(n_splits=n_splits, embargo=embargo)
    
    if verbose:
        print(f"🔄 Validação walk-forward: {n_splits} splits, embargo={embargo} dia(s)")
        print(f"   Métricas: {', '.join(metrics.keys())}")
    
    # Iterar pelos folds
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        # Clonar modelo para cada fold
        model_fold = clone(model)
        
        # Treinar
        model_fold.fit(X[train_idx], y[train_idx])
        
        # Prever
        if hasattr(model_fold, 'predict_proba'):
            y_pred = model_fold.predict_proba(X[test_idx])[:, 1]
        else:
            y_pred = model_fold.predict(X[test_idx])
        
        # Calcular métricas
        for metric_name, metric_func in metrics.items():
            try:
                score = metric_func(y[test_idx], y_pred)
                scores[metric_name].append(score)
                
                if verbose:
                    print(f"   Fold {fold}: {metric_name}={score:.4f}")
            except Exception as e:
                if verbose:
                    print(f"   Fold {fold}: {metric_name} falhou - {e}")
                scores[metric_name].append(np.nan)
    
    # Calcular estatísticas
    results = {
        'scores': scores,
        'mean': {name: np.mean(vals) for name, vals in scores.items()},
        'std': {name: np.std(vals) for name, vals in scores.items()}
    }
    
    if verbose:
        print("\n📊 Resultados finais:")
        for name in metrics.keys():
            print(f"   {name}: {results['mean'][name]:.4f} ± {results['std'][name]:.4f}")
    
    return results
