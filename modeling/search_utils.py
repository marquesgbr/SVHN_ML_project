"""
Hyperparameter Search Utilities
Contains functions for performing hyperparameter searches and visualizing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Constante para validação cruzada estratificada padrão
DEFAULT_CV_STRATEGY = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
SEARCHES_FOLDER = 'searches'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'

def plot_search_history(all_search_results, search_results, model_name, metric='mean_test_score'):
    """Plota a evolução dos resultados durante a busca de hiperparâmetros"""
        
    plt.figure(figsize=(15, 6))
        
    results_df = pd.DataFrame(search_results.cv_results_)
    # Extrair melhor score de cada busca e seu desvio-padrão
    search_scores = []
    search_stds = []
    search_indices = []
    
    for i, search_result in enumerate(all_search_results):
        search_scores.append(search_result['best_score'])
        search_indices.append(i + 1)
        
        # Encontrar o desvio-padrão correspondente ao melhor score desta busca
        cv_results = search_result['cv_results']
        best_idx = np.argmax(cv_results['mean_test_score'])
        search_stds.append(cv_results['std_test_score'][best_idx])
    
    # GRÁFICO 1: Melhor F1-Score por Busca com Desvio-Padrão
    plt.subplot(1, 2, 1)
    plt.plot(search_indices, search_scores, 'b-o', alpha=0.8, markersize=8)
    
    # Adicionar sombra do desvio-padrão
    plt.fill_between(search_indices, 
                     np.array(search_scores) - np.array(search_stds),
                     np.array(search_scores) + np.array(search_stds), 
                     color='blue', alpha=0.3)
    
    plt.title(f'{model_name} - Melhor F1-Score por Busca (com Desvio-Padrão)')
    plt.xlabel('Número da Busca')
    plt.ylabel('Melhor F1-Score')
    plt.grid(True, alpha=0.3)
    
    # Destacar a melhor busca
    best_idx = search_scores.index(max(search_scores))
    plt.plot(search_indices[best_idx], search_scores[best_idx], 'ro', markersize=12, 
            markeredgecolor='darkred', markeredgewidth=2,
            label=f'Melhor: {search_scores[best_idx]:.4f} ± {search_stds[best_idx]:.4f}')
    plt.legend()
    
    # GRÁFICO 2: Iterações da melhor busca
    plt.subplot(1, 2, 2)
    
    # Verificar se existe coluna de treino
    if 'mean_train_score' in results_df.columns:
        plt.plot(results_df['mean_train_score'], 'g-o', alpha=0.7, label='Treino')
    
    # Plotar validação
    plt.plot(results_df[metric], 'b-o', alpha=0.7, label='Validação')
    
    # DESTACAR A MELHOR ITERAÇÃO
    best_iteration_idx = results_df[metric].idxmax()
    best_iteration_score = results_df[metric].iloc[best_iteration_idx]
    
    plt.plot(best_iteration_idx, best_iteration_score, 'ro', markersize=15, 
            markeredgecolor='darkred', markeredgewidth=2,
            label=f'Melhor iteração: #{best_iteration_idx + 1} ({best_iteration_score:.4f})')
    
    plt.title(f'{model_name} - Treino vs Validação (Melhor Busca)')
    plt.xlabel('Iteração')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def multiple_randomized_search(estimator, param_distributions, X, y, cv_strategy, 
                              n_searches=20, n_iter_per_search=80, scoring='f1', 
                              random_state=42, n_jobs=-1, verbose=0):
    """
    Executa múltiplas buscas RandomizedSearchCV e retorna a melhor configuração global
    
    Parameters:
    -----------
    n_searches : int
        Número de execuções do RandomizedSearchCV (default: 20)
    n_iter_per_search : int  
        Número de iterações por execução (default: 80)
    """
    print(f"Executando {n_searches} buscas com {n_iter_per_search} iterações cada...")
    
    best_overall_score = -np.inf
    best_overall_params = None
    best_search_result = None
    all_results = []
    
    for search_idx in range(n_searches):
        print(f"\nBusca {search_idx + 1}/{n_searches}...")
        
        # RandomizedSearchCV para esta execução
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter_per_search,
            scoring=scoring,
            cv=cv_strategy,
            random_state=None,
            n_jobs=n_jobs,
            return_train_score=True,
            verbose=0  # Menos verbose para múltiplas execuções
        )
        
        search.fit(X, y)
        
        # Armazenar resultados desta busca
        search_results = {
            'search_idx': search_idx,
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        all_results.append(search_results)
        
        # Verificar se esta é a melhor busca até agora
        if search.best_score_ > best_overall_score:
            best_overall_score = search.best_score_
            best_overall_params = search.best_params_
            best_search_result = search
            
        print(f"Melhor score desta busca: {search.best_score_:.4f}")
        print(f"Melhor configuração desta busca: {search.best_params_}")
        print(f"Melhor score geral até agora: {best_overall_score:.4f}")
    
    print(f"\n🎯 Busca completa! Melhor score geral: {best_overall_score:.4f}")
    print(f"Total de configurações testadas: {n_searches * n_iter_per_search:,}")
    
    return best_search_result, all_results, best_overall_params


def plot_search_history_from_loaded(loaded_results, model_name, metric='mean_test_score'):
    """Plota a evolução dos resultados a partir de loaded_results"""
    
    plt.figure(figsize=(15, 6))
    
    # Extrair dados do loaded_results
    detailed_df = loaded_results['detailed_df']
    
    # Agrupar por search_idx para obter o melhor de cada busca e seu desvio-padrão
    best_per_search = detailed_df.loc[detailed_df.groupby('search_idx')['mean_test_score'].idxmax()]
    best_per_search = best_per_search[['search_idx', 'mean_test_score', 'std_test_score']].reset_index(drop=True)
    best_per_search['search_number'] = best_per_search['search_idx'] + 1  # 1-indexado
    
    # GRÁFICO 1: Melhor F1-Score por Busca com Desvio-Padrão
    plt.subplot(1, 2, 1)
    plt.plot(best_per_search['search_number'], best_per_search['mean_test_score'], 
             'b-o', alpha=0.8, markersize=8)
    
    # Adicionar sombra do desvio-padrão
    plt.fill_between(best_per_search['search_number'], 
                     best_per_search['mean_test_score'] - best_per_search['std_test_score'],
                     best_per_search['mean_test_score'] + best_per_search['std_test_score'], 
                     color='blue', alpha=0.3)
    
    plt.title(f'{model_name} - Melhor F1-Score por Busca (com Desvio-Padrão)')
    plt.xlabel('Número da Busca')
    plt.ylabel('Melhor F1-Score')
    plt.grid(True, alpha=0.3)
    
    # Destacar a melhor busca
    best_search_idx = best_per_search['mean_test_score'].idxmax()
    best_search_score = best_per_search['mean_test_score'].iloc[best_search_idx]
    best_search_std = best_per_search['std_test_score'].iloc[best_search_idx]
    best_search_number = best_per_search['search_number'].iloc[best_search_idx]
    
    plt.plot(best_search_number, best_search_score, 'ro', markersize=12, 
             markeredgecolor='darkred', markeredgewidth=2,
             label=f'Melhor: {best_search_score:.4f} ± {best_search_std:.4f}')
    plt.legend()
    
    # GRÁFICO 2: Iterações da melhor busca
    plt.subplot(1, 2, 2)
    
    # Encontrar qual search_idx teve o melhor score geral
    best_overall_idx = detailed_df['mean_test_score'].idxmax()
    best_overall_search_idx = detailed_df.loc[best_overall_idx, 'search_idx']
    
    # Filtrar dados apenas da melhor busca
    best_search_data = detailed_df[detailed_df['search_idx'] == best_overall_search_idx].copy()
    best_search_data = best_search_data.sort_values('iteration').reset_index(drop=True)
    
    # Criar eixo X 1-indexado para iterações
    iterations_1indexed = range(1, len(best_search_data) + 1)
    
    # Verificar se existe coluna de treino
    if 'mean_train_score' in best_search_data.columns and best_search_data['mean_train_score'].notna().any():
        plt.plot(iterations_1indexed, best_search_data['mean_train_score'], 
                'g-o', alpha=0.7, label='Treino')
    
    # Plotar validação sem sombra
    plt.plot(iterations_1indexed, best_search_data[metric], 'b-o', alpha=0.7, label='Validação')
    
    # DESTACAR A MELHOR ITERAÇÃO
    best_iteration_idx = best_search_data[metric].idxmax()
    best_iteration_score = best_search_data[metric].iloc[best_iteration_idx]
    best_iteration_number = best_iteration_idx + 1  # 1-indexado
    
    plt.plot(best_iteration_number, best_iteration_score, 'ro', markersize=15, 
             markeredgecolor='darkred', markeredgewidth=2,
             label=f'Melhor iteração: #{best_iteration_number} ({best_iteration_score:.4f})')
    
    plt.title(f'{model_name} - Treino vs Validação (Melhor Busca #{best_overall_search_idx + 1})')
    plt.xlabel('Iteração')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def load_search_results(model_name, searches_folder='searches'):
    """
    Carrega resultados de busca salvos anteriormente
    
    Returns:
    --------
    dict: Dicionário com todos os resultados carregados
    """
    print(f"=== CARREGANDO RESULTADOS DE BUSCA - {model_name.upper()} ===")
    
    results = {}
    
    # 1. Carregar DataFrame detalhado
    csv_path = os.path.join(searches_folder, f'{model_name.lower()}_all_searches.csv')
    if os.path.exists(csv_path):
        results['detailed_df'] = pd.read_csv(csv_path)
        print(f"✅ Resultados detalhados carregados: {len(results['detailed_df']):,} configurações")
    else:
        print(f"⚠️  Arquivo não encontrado: {csv_path}")
    
    # 2. Carregar resumo JSON
    json_path = os.path.join(searches_folder, f'{model_name.lower()}_search_summary.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results['summary'] = json.load(f)
        print(f"✅ Resumo carregado: F1-Score = {results['summary']['best_overall_score']:.4f}")
    else:
        print(f"⚠️  Arquivo não encontrado: {json_path}")
    
    return results


def get_best_params_from_saved(model_name, searches_folder='searches'):
    """
    Recupera os melhores parâmetros de arquivos salvos
    
    Returns:
    --------
    dict: Melhores parâmetros encontrados
    """
    # Tentar carregar do JSON primeiro
    json_path = os.path.join(searches_folder, f'{model_name.lower()}_search_summary.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            summary = json.load(f)
        return summary['best_overall_params']
    
    print(f"❌ Não foi possível carregar parâmetros para {model_name}")
    return None


def save_search_results(model_name, model_search, model_all_searches, 
                        n_searches, n_iter_per_search, scoring='f1', cv_folds=5,
                        top_params_columns=None, searches_folder='searches'):
    """
    Salva os resultados da busca de hiperparâmetros
    
    Parameters:
    -----------
    model_name : str
        Nome do modelo
    model_search : RandomizedSearchCV
        Objeto do melhor resultado de busca
    model_all_searches : list
        Lista com todos os resultados de busca
    n_searches : int
        Número de buscas realizadas
    n_iter_per_search : int
        Número de iterações por busca
    scoring : str
        Métrica de scoring usada
    cv_folds : int
        Número de folds da validação cruzada
    top_params_columns : list
        Lista de colunas de parâmetros para incluir no top configs (opcional)
    searches_folder : str
        Pasta onde salvar os resultados
    
    Returns:
    --------
    pd.DataFrame: DataFrame com todos os resultados detalhados
    """
    print(f"=== SALVANDO RESULTADOS DA BUSCA - {model_name.upper()} ===")
    
    # Criar pasta se não existir
    os.makedirs(searches_folder, exist_ok=True)
    
    # 1. Salvar resultados detalhados de todas as buscas
    search_detailed_results = []
    
    for i, search_result in enumerate(model_all_searches):
        # Extrair informações de cada busca individual
        cv_results = search_result['cv_results']
        
        for j in range(len(cv_results['mean_test_score'])):
            result_dict = {
                'search_idx': search_result['search_idx'],
                'iteration': j,
                'mean_test_score': cv_results['mean_test_score'][j],
                'std_test_score': cv_results['std_test_score'][j],
                'mean_train_score': cv_results['mean_train_score'][j] if 'mean_train_score' in cv_results else None,
                'std_train_score': cv_results['std_train_score'][j] if 'std_train_score' in cv_results else None,
                'params': str(cv_results['params'][j]),
            }
            # Adicionar parâmetros individuais
            result_dict.update({k: (str(v) if isinstance(v, tuple) else v) 
                               for k, v in cv_results['params'][j].items()})
            search_detailed_results.append(result_dict)
    
    # Converter para DataFrame e salvar
    search_df = pd.DataFrame(search_detailed_results)
    csv_path = os.path.join(searches_folder, f'{model_name.lower()}_all_searches.csv')
    search_df.to_csv(csv_path, index=False)
    
    print(f"  ✅ Todos os Resultados salvos: {csv_path}")
    print(f"  Total de configurações testadas: {len(search_df):,}")
    
    # 2. Salvar resumo da melhor busca
    best_search_summary = {
        'model_name': model_name,
        'best_overall_score': model_search.best_score_,
        'best_overall_params': {k: (str(v) if isinstance(v, tuple) else v) 
                               for k, v in model_search.best_params_.items()},
        'search_config': {
            'n_searches': n_searches,
            'n_iter_per_search': n_iter_per_search,
            'scoring': scoring,
            'cv_folds': cv_folds,
            'total_configurations': len(search_df)
        }
    }
    
    # Adicionar top configs se colunas especificadas
    if top_params_columns:
        available_cols = ['mean_test_score', 'std_test_score'] + [
            col for col in top_params_columns if col in search_df.columns
        ]
        best_search_summary['top_configs'] = search_df.nlargest(10, 'mean_test_score')[
            available_cols
        ].to_dict('records')
    
    # Salvar resumo em JSON
    json_path = os.path.join(searches_folder, f'{model_name.lower()}_search_summary.json')
    with open(json_path, 'w') as f:
        json.dump(best_search_summary, f, indent=2)
    
    print(f"  ✅ Resumo salvo: {json_path}")
    
    # Mostrar estatísticas da busca
    print(f"\n--- ESTATÍSTICAS DA BUSCA {model_name.upper()} ---")
    print(f"Melhor F1-Score: {model_search.best_score_:.4f}")
    print(f"Desvio padrão do melhor: {search_df.loc[search_df['mean_test_score'].idxmax(), 'std_test_score']:.4f}")
    print(f"F1-Score médio geral: {search_df['mean_test_score'].mean():.4f}")
    print(f"F1-Score mínimo: {search_df['mean_test_score'].min():.4f}")
    print(f"F1-Score máximo: {search_df['mean_test_score'].max():.4f}")
    
    return search_df


def save_final_results(model_name, best_params, best_score, train_metrics, 
                       test_metrics, y_pred, y_test, X_train_scaled, X_test_scaled,
                       results_folder='results'):
    """
    Salva os resultados finais da avaliação do modelo
    
    Parameters:
    -----------
    model_name : str
        Nome do modelo
    best_params : dict
        Melhores parâmetros encontrados
    best_score : float
        Melhor score da validação cruzada
    train_metrics : dict
        Métricas de treino
    test_metrics : dict
        Métricas de teste
    y_pred : array
        Predições do modelo
    y_test : array
        Labels verdadeiros de teste
    X_train_scaled : array
        Dataset de treino (para contagem)
    X_test_scaled : array
        Dataset de teste (para contagem)
    results_folder : str
        Pasta onde salvar os resultados
    
    Returns:
    --------
    dict: Dicionário com todos os resultados
    """
    # Criar pasta se não existir
    os.makedirs(results_folder, exist_ok=True)
    
    # Compilar todos os resultados
    model_final_results = {
        'model_name': model_name,
        'best_params': {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        'best_cv_score': best_score,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'predictions': y_pred.tolist(),
        'test_labels': y_test.tolist(),
        'evaluation_info': {
            'train_samples_used': len(X_train_scaled),
            'test_samples_used': len(X_test_scaled),
            'total_train_samples': len(X_train_scaled),
            'total_test_samples': len(X_test_scaled)
        }
    }
    
    # Salvar resultados em JSON
    json_path = os.path.join(results_folder, f'{model_name.lower()}_results.json')
    with open(json_path, 'w') as f:
        json.dump(model_final_results, f, indent=2)
    
    print(f"✅ Resultados {model_name} salvos em: {json_path}")
    
    return model_final_results
