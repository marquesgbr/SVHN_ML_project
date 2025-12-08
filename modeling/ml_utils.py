"""
Machine Learning Utilities for Model Evaluation
Contains functions for calculating metrics and evaluating models
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)


def gmean_score(y_true, y_pred):
    """Calcula o G-Mean (Geometric Mean) para problemas multiclasse"""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0) 
    return np.prod(recalls) ** (1 / len(recalls))


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Avalia um modelo treinado e retorna métricas completas"""
    # Predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas de treino
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='macro', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='macro', zero_division=0),
        'f1': f1_score(y_train, y_train_pred, average='macro', zero_division=0),
        'gmean': gmean_score(y_train, y_train_pred)
    }
    
    # Métricas de teste
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
        'gmean': gmean_score(y_test, y_test_pred)
    }
    
    return train_metrics, test_metrics, y_test_pred


def load_and_prepare_datasets(train_path='../SVHN_train_prep.csv',
                               test_path='../SVHN_test_prep.csv',
                               target_column='digit',
                               scale=True):
    """
    Carrega e prepara os datasets de treino e teste
    
    Parameters:
    -----------
    train_path : str
        Caminho para o dataset de treino
    test_path : str
        Caminho para o dataset de teste
    target_column : str
        Nome da coluna target
    scale : bool
        Se True, aplica StandardScaler nos dados
    
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, train_data, test_data, scaler)
    """
    # Carregar datasets pré-processados
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separar features e target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    # Normalização dos dados
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, train_data, test_data, scaler
