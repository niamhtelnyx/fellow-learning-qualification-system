#!/usr/bin/env python3
"""
ML Model Training for Fellow Learning Qualification System
Trains interpretable models to predict AE progression and qualification scores
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main class for training qualification models"""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Model configurations
        self.model_configs = {
            'logistic': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state, n_estimators=100),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state, objective='binary:logistic'),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'gradient_boost': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        self.trained_models = {}
        self.model_performance = {}
        
    def train_progression_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                               feature_names: List[str] = None) -> Dict[str, Any]:
        """Train binary classification model for AE progression prediction"""
        logger.info("Training AE progression prediction models")
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name} model...")
            
            try:
                # Grid search for best parameters
                grid_search = GridSearchCV(
                    config['model'], 
                    config['param_grid'],
                    cv=3,  # 3-fold CV for small datasets
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Calibrate model for probability outputs
                calibrated_model = CalibratedClassifierCV(best_model, cv=3)
                calibrated_model.fit(X_train, y_train)
                
                # Cross-validation scores
                cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=3, scoring='roc_auc')
                
                # Store model and results
                self.trained_models[f'progression_{model_name}'] = calibrated_model
                
                results[model_name] = {
                    'model': calibrated_model,
                    'best_params': grid_search.best_params_,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'best_cv_score': grid_search.best_score_
                }
                
                # Feature importance (if available)
                if hasattr(best_model, 'feature_importances_') and feature_names:
                    importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                    # Sort by importance
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    results[model_name]['feature_importance'] = sorted_importance[:10]  # Top 10
                
                logger.info(f"{model_name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def train_voice_ai_model(self, X_train: np.ndarray, y_voice_ai: np.ndarray,
                            feature_names: List[str] = None) -> Dict[str, Any]:
        """Train specialized model for Voice AI prospect identification"""
        logger.info("Training Voice AI identification models")
        
        # Use only the best performing models for Voice AI
        voice_ai_configs = {
            'xgboost': self.model_configs['xgboost'],
            'random_forest': self.model_configs['random_forest']
        }
        
        results = {}
        
        for model_name, config in voice_ai_configs.items():
            try:
                # Grid search
                grid_search = GridSearchCV(
                    config['model'],
                    config['param_grid'],
                    cv=3,
                    scoring='precision',  # Focus on precision for Voice AI
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_voice_ai)
                best_model = grid_search.best_estimator_
                
                # Calibrate model
                calibrated_model = CalibratedClassifierCV(best_model, cv=3)
                calibrated_model.fit(X_train, y_voice_ai)
                
                # Cross-validation scores
                cv_scores = cross_val_score(calibrated_model, X_train, y_voice_ai, cv=3, scoring='precision')
                
                # Store model
                self.trained_models[f'voice_ai_{model_name}'] = calibrated_model
                
                results[model_name] = {
                    'model': calibrated_model,
                    'best_params': grid_search.best_params_,
                    'cv_precision_mean': cv_scores.mean(),
                    'cv_precision_std': cv_scores.std(),
                    'best_cv_score': grid_search.best_score_
                }
                
                logger.info(f"Voice AI {model_name} - CV Precision: {cv_scores.mean():.3f}")
                
            except Exception as e:
                logger.error(f"Error training Voice AI {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       y_voice_ai_test: np.ndarray = None) -> Dict[str, Dict]:
        """Evaluate all trained models on test set"""
        logger.info("Evaluating trained models")
        
        evaluation_results = {}
        
        # Evaluate progression models
        for model_name, model in self.trained_models.items():
            if 'progression' in model_name:
                try:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'auc_roc': roc_auc_score(y_test, y_pred_proba),
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                    }
                    
                    evaluation_results[model_name] = results
                    
                    logger.info(f"{model_name} - Accuracy: {results['accuracy']:.3f}, "
                              f"Precision: {results['precision']:.3f}, "
                              f"Recall: {results['recall']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    evaluation_results[model_name] = {'error': str(e)}
        
        # Evaluate Voice AI models if labels provided
        if y_voice_ai_test is not None:
            for model_name, model in self.trained_models.items():
                if 'voice_ai' in model_name:
                    try:
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        results = {
                            'accuracy': accuracy_score(y_voice_ai_test, y_pred),
                            'precision': precision_score(y_voice_ai_test, y_pred),
                            'recall': recall_score(y_voice_ai_test, y_pred),
                            'f1_score': f1_score(y_voice_ai_test, y_pred),
                            'auc_roc': roc_auc_score(y_voice_ai_test, y_pred_proba),
                            'confusion_matrix': confusion_matrix(y_voice_ai_test, y_pred).tolist()
                        }
                        
                        evaluation_results[model_name] = results
                        
                    except Exception as e:
                        logger.error(f"Error evaluating Voice AI {model_name}: {e}")
                        evaluation_results[model_name] = {'error': str(e)}
        
        self.model_performance = evaluation_results
        return evaluation_results
    
    def select_best_models(self) -> Dict[str, str]:
        """Select best performing models for each task"""
        best_models = {}
        
        # Best progression model
        best_progression_score = 0
        best_progression_model = None
        
        for model_name, performance in self.model_performance.items():
            if 'progression' in model_name and 'error' not in performance:
                # Weighted score: accuracy + precision + recall + auc
                score = (performance['accuracy'] + performance['precision'] + 
                        performance['recall'] + performance['auc_roc']) / 4
                
                if score > best_progression_score:
                    best_progression_score = score
                    best_progression_model = model_name
        
        if best_progression_model:
            best_models['progression'] = best_progression_model
            logger.info(f"Best progression model: {best_progression_model} (score: {best_progression_score:.3f})")
        
        # Best Voice AI model
        best_voice_ai_score = 0
        best_voice_ai_model = None
        
        for model_name, performance in self.model_performance.items():
            if 'voice_ai' in model_name and 'error' not in performance:
                # Focus on precision for Voice AI
                score = performance['precision']
                
                if score > best_voice_ai_score:
                    best_voice_ai_score = score
                    best_voice_ai_model = model_name
        
        if best_voice_ai_model:
            best_models['voice_ai'] = best_voice_ai_model
            logger.info(f"Best Voice AI model: {best_voice_ai_model} (precision: {best_voice_ai_score:.3f})")
        
        return best_models
    
    def save_models(self, model_version: str = None) -> str:
        """Save trained models and metadata"""
        if not model_version:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        version_dir = self.model_dir / f"v_{model_version}"
        version_dir.mkdir(exist_ok=True)
        
        # Save all trained models
        models_saved = []
        for model_name, model in self.trained_models.items():
            model_path = version_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            models_saved.append(model_name)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save performance metrics
        performance_path = version_dir / "performance_metrics.json"
        with open(performance_path, 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        
        # Save model metadata
        metadata = {
            'version': model_version,
            'created_at': datetime.now().isoformat(),
            'models_trained': models_saved,
            'best_models': self.select_best_models(),
            'model_dir': str(version_dir)
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model version {model_version} saved to {version_dir}")
        return model_version
    
    def load_models(self, model_version: str) -> Dict[str, Any]:
        """Load trained models from disk"""
        version_dir = self.model_dir / f"v_{model_version}"
        
        if not version_dir.exists():
            raise ValueError(f"Model version {model_version} not found")
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load models
        loaded_models = {}
        for model_name in metadata['models_trained']:
            model_path = version_dir / f"{model_name}.joblib"
            if model_path.exists():
                loaded_models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name}")
        
        self.trained_models = loaded_models
        
        # Load performance metrics
        performance_path = version_dir / "performance_metrics.json"
        if performance_path.exists():
            with open(performance_path, 'r') as f:
                self.model_performance = json.load(f)
        
        logger.info(f"Loaded model version {model_version}")
        return metadata

class QualificationScorer:
    """Generate qualification scores using trained models"""
    
    def __init__(self, progression_model: Any, voice_ai_model: Any = None):
        self.progression_model = progression_model
        self.voice_ai_model = voice_ai_model
    
    def score_lead(self, features: np.ndarray, company_name: str = "Unknown") -> Dict:
        """Generate qualification score for a single lead"""
        
        # Get progression probability
        progression_prob = self.progression_model.predict_proba(features.reshape(1, -1))[0, 1]
        progression_pred = self.progression_model.predict(features.reshape(1, -1))[0]
        
        # Get Voice AI probability if model available
        voice_ai_prob = 0.0
        voice_ai_pred = 0
        if self.voice_ai_model:
            voice_ai_prob = self.voice_ai_model.predict_proba(features.reshape(1, -1))[0, 1]
            voice_ai_pred = self.voice_ai_model.predict(features.reshape(1, -1))[0]
        
        # Calculate qualification score (0-100)
        base_score = progression_prob * 100
        
        # Boost for Voice AI prospects
        if voice_ai_pred == 1:
            base_score = min(100, base_score * 1.2)  # 20% boost
        
        qualification_score = int(np.round(base_score))
        
        # Determine routing recommendation
        if qualification_score >= 85:
            recommendation = "AE_HANDOFF"
            priority = "HIGH_VOICE_AI" if voice_ai_pred == 1 else "HIGH"
        elif qualification_score >= 70:
            recommendation = "AE_HANDOFF" 
            priority = "MEDIUM"
        elif qualification_score >= 50:
            recommendation = "NURTURE_TRACK"
            priority = "LOW"
        else:
            recommendation = "SELF_SERVICE"
            priority = "VERY_LOW"
        
        # Generate reasoning
        reasoning = []
        if progression_prob > 0.7:
            reasoning.append("High likelihood of AE progression")
        if voice_ai_pred == 1:
            reasoning.append("Strong Voice AI prospect signals")
        if qualification_score < 50:
            reasoning.append("Limited qualification signals detected")
        
        return {
            "company_name": company_name,
            "qualification_score": qualification_score,
            "voice_ai_fit": int(voice_ai_prob * 100),
            "progression_probability": float(progression_prob),
            "recommendation": recommendation,
            "reasoning": reasoning,
            "priority": priority,
            "confidence": max(progression_prob, 1 - progression_prob),  # Distance from 0.5
            "model_version": "baseline_v1"
        }

def train_baseline_model():
    """Train baseline models using sample data"""
    from feature_engineer import FeatureEngineeringPipeline, load_sample_data
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Training baseline qualification models")
    
    # Load and prepare data
    call_data, company_data = load_sample_data()
    
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.prepare_training_data(call_data, company_data)
    features_df = pipeline.create_target_labels(features_df)
    
    # Transform features
    X = pipeline.fit_transform(features_df)
    y_progression = features_df['target_progression'].values
    y_voice_ai = features_df['target_voice_ai_fit'].values
    
    # Split data (small dataset, so minimal split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_progression, test_size=0.2, random_state=42, stratify=y_progression
    )
    y_voice_ai_train = y_voice_ai[:-1]  # Corresponding to train split
    y_voice_ai_test = y_voice_ai[-1:]   # Corresponding to test split
    
    # Train models
    trainer = ModelTrainer()
    
    # Train progression models
    progression_results = trainer.train_progression_model(
        X_train, y_train, pipeline.feature_columns
    )
    
    # Train Voice AI models  
    voice_ai_results = trainer.train_voice_ai_model(
        X_train, y_voice_ai_train, pipeline.feature_columns
    )
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test, y_voice_ai_test)
    
    # Save models
    model_version = trainer.save_models("baseline")
    
    logger.info("Baseline model training complete!")
    logger.info(f"Model version: {model_version}")
    
    # Test qualification scoring
    best_models = trainer.select_best_models()
    if best_models.get('progression'):
        progression_model = trainer.trained_models[best_models['progression']]
        voice_ai_model = trainer.trained_models.get(best_models.get('voice_ai'))
        
        scorer = QualificationScorer(progression_model, voice_ai_model)
        
        # Score test samples
        for i, features in enumerate(X_test):
            company_name = ["Test Company A", "Test Company B"][i % 2]
            score_result = scorer.score_lead(features, company_name)
            
            print(f"\nQualification Result for {company_name}:")
            print(f"  Score: {score_result['qualification_score']}")
            print(f"  Voice AI Fit: {score_result['voice_ai_fit']}")
            print(f"  Recommendation: {score_result['recommendation']}")
            print(f"  Priority: {score_result['priority']}")
            print(f"  Reasoning: {', '.join(score_result['reasoning'])}")
    
    return model_version

if __name__ == "__main__":
    train_baseline_model()