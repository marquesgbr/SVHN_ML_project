# ======================================================================
# COMITÊ DE REDES NEURAIS - IMPLEMENTAÇÃO (ESTILO STACKING)
# ======================================================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

class NeuralNetworkCommittee(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 # Parâmetros da MLP 1
                 mlp1_hidden_layers=(100,),
                 mlp1_alpha=0.01,
                 mlp1_learning_rate_init=0.001,
                 mlp1_max_iter=200,
                 mlp1_activation='relu',
                 mlp1_learning_rate='constant',
                 
                 # Parâmetros da MLP 2
                 mlp2_hidden_layers=(100, 50),
                 mlp2_alpha=0.01,
                 mlp2_learning_rate_init=0.001,
                 mlp2_max_iter=200,
                 mlp2_activation='tanh',
                 mlp2_learning_rate='adaptive',
                 
                 # Parâmetros da MLP 3
                 mlp3_hidden_layers=(50,),
                 mlp3_alpha=0.01,
                 mlp3_learning_rate_init=0.001,
                 mlp3_max_iter=200,
                 mlp3_activation='logistic',
                 mlp3_learning_rate='constant',
                 
                 # Parâmetros do comitê
                 solver='lbfgs', 
                 voting='soft',
                 random_state=None):
        """
        Comitê de 3 Redes Neurais com parâmetros individuais (estilo Stacking)
        
        Cada rede do comitê tem seus próprios hiperparâmetros otimizados
        independentemente, permitindo configurações heterogêneas.
        
        Parameters:
        -----------
        mlp1_* : diversos
            Hiperparâmetros da primeira rede neural
        mlp2_* : diversos
            Hiperparâmetros da segunda rede neural
        mlp3_* : diversos
            Hiperparâmetros da terceira rede neural
        solver : str
            Algoritmo de otimização (comum a todas)
        voting : str
            Tipo de votação ('hard' ou 'soft')
        random_state : int
            Seed para reprodutibilidade
        """
        # MLP 1
        self.mlp1_hidden_layers = mlp1_hidden_layers
        self.mlp1_alpha = mlp1_alpha
        self.mlp1_learning_rate_init = mlp1_learning_rate_init
        self.mlp1_max_iter = mlp1_max_iter
        self.mlp1_activation = mlp1_activation
        self.mlp1_learning_rate = mlp1_learning_rate
        
        # MLP 2
        self.mlp2_hidden_layers = mlp2_hidden_layers
        self.mlp2_alpha = mlp2_alpha
        self.mlp2_learning_rate_init = mlp2_learning_rate_init
        self.mlp2_max_iter = mlp2_max_iter
        self.mlp2_activation = mlp2_activation
        self.mlp2_learning_rate = mlp2_learning_rate
        
        # MLP 3
        self.mlp3_hidden_layers = mlp3_hidden_layers
        self.mlp3_alpha = mlp3_alpha
        self.mlp3_learning_rate_init = mlp3_learning_rate_init
        self.mlp3_max_iter = mlp3_max_iter
        self.mlp3_activation = mlp3_activation
        self.mlp3_learning_rate = mlp3_learning_rate
        
        # Comitê
        self.solver = solver
        self.voting = voting
        self.random_state = random_state
        
    def fit(self, X, y):
        # Criar as 3 redes neurais com parâmetros individuais
        mlp_1 = MLPClassifier(
            hidden_layer_sizes=self.mlp1_hidden_layers,
            alpha=self.mlp1_alpha,
            learning_rate_init=self.mlp1_learning_rate_init,
            max_iter=self.mlp1_max_iter,
            activation=self.mlp1_activation,
            solver=self.solver,
            random_state=self.random_state,
            early_stopping=True,
            learning_rate=self.mlp1_learning_rate,
            validation_fraction=0.1
        )
        
        mlp_2 = MLPClassifier(
            hidden_layer_sizes=self.mlp2_hidden_layers,
            alpha=self.mlp2_alpha,
            learning_rate_init=self.mlp2_learning_rate_init,
            max_iter=self.mlp2_max_iter,
            activation=self.mlp2_activation,
            solver=self.solver,
            random_state=self.random_state + 1 if self.random_state is not None else None,
            early_stopping=True,
            learning_rate=self.mlp2_learning_rate,
            validation_fraction=0.1
        )
        
        mlp_3 = MLPClassifier(
            hidden_layer_sizes=self.mlp3_hidden_layers,
            alpha=self.mlp3_alpha,
            learning_rate_init=self.mlp3_learning_rate_init,
            max_iter=self.mlp3_max_iter,
            activation=self.mlp3_activation,
            solver=self.solver,
            random_state=self.random_state + 2 if self.random_state is not None else None,
            early_stopping=True,
            learning_rate=self.mlp3_learning_rate,
            validation_fraction=0.1
        )
        
        networks = [
            ('mlp_1', mlp_1),
            ('mlp_2', mlp_2),
            ('mlp_3', mlp_3)
        ]
        
        # Criar o comitê com VotingClassifier
        self.committee = VotingClassifier(
            estimators=networks,
            voting=self.voting,
            n_jobs=-1
        )
        
        # Treinar o comitê
        self.committee.fit(X, y)
        self.classes_ = self.committee.classes_
        
        return self
    
    def predict(self, X):
        return self.committee.predict(X)
    
    def predict_proba(self, X):
        if self.voting == 'soft':
            return self.committee.predict_proba(X)
        else:
            raise AttributeError("predict_proba is not available when voting='hard'")
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ======================================================================
# COMITÊ HETEROGÊNEO (STACKING) - IMPLEMENTAÇÃO COM WEAK LEARNERS
# ======================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb

class HeterogeneousStackingCommittee(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 # === DECISION TREE - Weak Learner ===
                 dt_max_depth=3,
                 dt_min_samples_split=20,
                 dt_min_samples_leaf=10,
                 dt_criterion='gini',
                 dt_max_features=None,
                 
                 # === MLP - Weak Learner ===
                 mlp_hidden_layers=(50,),
                 mlp_alpha=1.0,
                 mlp_learning_rate_init=0.01,
                 mlp_max_iter=100,
                 mlp_activation='relu',
                 mlp_learning_rate='constant',
                 
                 # === XGBOOST - Weak Learner ===
                 xgb_n_estimators=30,
                 xgb_max_depth=2,
                 xgb_learning_rate=0.05,
                 xgb_subsample=0.5,
                 xgb_colsample_bytree=0.5,
                 xgb_min_child_weight=5,
                 xgb_gamma=0.1,
                 xgb_reg_alpha=1.0,
                 xgb_reg_lambda=1.0,
                 
                 # === META-ESTIMADOR ===
                 meta_C=1.0,
                 meta_max_iter=1000,
                 
                 # === CONFIGURAÇÕES GERAIS ===
                 cv=5,
                 random_state=None):
        """
        Comitê Heterogêneo com Weak Learners Diversos
        
        Estimadores Base (Weak Learners):
        ----------------------------------
        - Decision Tree: Árvore rasa com regras simples
        - MLP: Rede neural shallow com regularização forte
        - XGBoost: Gradient boosting conservador
        
        Meta-Estimador:
        ---------------
        - Logistic Regression: Aprende combinação linear ótima
        
        Estratégia:
        -----------
        - Cada modelo é "fraco" individualmente 
        - Diversidade de paradigmas garante erros não-correlacionados
        - Stacking combina forças e corrige fraquezas
        """
        # Decision Tree
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.dt_criterion = dt_criterion
        self.dt_max_features = dt_max_features
        
        # MLP
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_alpha = mlp_alpha
        self.mlp_learning_rate_init = mlp_learning_rate_init
        self.mlp_max_iter = mlp_max_iter
        self.mlp_activation = mlp_activation
        self.mlp_learning_rate = mlp_learning_rate
        
        # XGBoost
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_gamma = xgb_gamma
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda
        
        # Meta-estimador
        self.meta_C = meta_C
        self.meta_max_iter = meta_max_iter
        
        # Geral
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, y):
        # Criar weak estimators
        base_estimators = [
            ('weak_dt', DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                min_samples_leaf=self.dt_min_samples_leaf,
                criterion=self.dt_criterion,
                max_features=self.dt_max_features,
                random_state=self.random_state
            )),
            ('weak_mlp', MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layers,
                alpha=self.mlp_alpha,
                learning_rate_init=self.mlp_learning_rate_init,
                learning_rate=self.mlp_learning_rate,
                max_iter=self.mlp_max_iter,
                activation=self.mlp_activation,
                solver='lbfgs',
                early_stopping=True,
                validation_fraction=0.2,
                random_state=self.random_state,
                verbose=False
            )),
            ('weak_xgb', xgb.XGBClassifier(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                subsample=self.xgb_subsample,
                colsample_bytree=self.xgb_colsample_bytree,
                min_child_weight=self.xgb_min_child_weight,
                gamma=self.xgb_gamma,
                reg_alpha=self.xgb_reg_alpha,
                reg_lambda=self.xgb_reg_lambda,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=self.random_state,
                verbosity=0
            ))
        ]
        
        # Meta-estimador
        meta_estimator = LogisticRegression(
            C=self.meta_C,
            max_iter=self.meta_max_iter,
            random_state=self.random_state
        )
        
        # Criar Stacking
        self.stacking_classifier = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=self.cv,
            stack_method='predict_proba',
            n_jobs=1
        )
        
        # Treinar
        self.stacking_classifier.fit(X, y)
        self.classes_ = self.stacking_classifier.classes_
        
        return self
    
    def predict(self, X):
        return self.stacking_classifier.predict(X)
    
    def predict_proba(self, X):
        return self.stacking_classifier.predict_proba(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# Implementação da classe LVQ (Learning Vector Quantization)
class LVQClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, prototypes_per_class=10, n_epochs=100, learning_rate=0.1, random_state=None):
        self.prototypes_per_class = prototypes_per_class
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Inicializar protótipos
        self.prototypes_ = []
        self.prototype_labels_ = []
        
        for class_label in self.classes_:
            class_data = X[y == class_label]
            n_prototypes = min(self.prototypes_per_class, len(class_data))
            
            # Inicializar protótipos aleatoriamente a partir dos dados da classe
            indices = np.random.choice(len(class_data), n_prototypes, replace=False)
            class_prototypes = class_data[indices].copy()
            
            self.prototypes_.extend(class_prototypes)
            self.prototype_labels_.extend([class_label] * n_prototypes)
        
        self.prototypes_ = np.array(self.prototypes_)
        self.prototype_labels_ = np.array(self.prototype_labels_)
        
        # Treinamento LVQ
        for epoch in range(self.n_epochs):
            for i, (sample, label) in enumerate(zip(X, y)):
                # Encontrar protótipo mais próximo
                distances = np.sum((self.prototypes_ - sample) ** 2, axis=1)
                winner_idx = np.argmin(distances)
                winner_label = self.prototype_labels_[winner_idx]
                
                # Atualizar protótipo
                if winner_label == label:
                    # Aproximar protótipo do exemplo (recompensa)
                    self.prototypes_[winner_idx] += self.learning_rate * (sample - self.prototypes_[winner_idx])
                else:
                    # Afastar protótipo do exemplo (punição)
                    self.prototypes_[winner_idx] -= self.learning_rate * (sample - self.prototypes_[winner_idx])
        
        return self
    
    def predict(self, X):
        predictions = []
        
        for sample in X:
            distances = np.sum((self.prototypes_ - sample) ** 2, axis=1)
            winner_idx = np.argmin(distances)
            predictions.append(self.prototype_labels_[winner_idx])
        
        return np.array(predictions)
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
