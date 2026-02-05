#!/usr/bin/env python3
"""
Continuous Learning System for Fellow Learning Qualification Model
Automatically retrains models with new Fellow call outcomes and monitors performance
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Local imports
from feature_engineer import FeatureEngineeringPipeline
from model_trainer import ModelTrainer, QualificationScorer

logger = logging.getLogger(__name__)

class ModelDriftDetector:
    """Detect model performance drift over time"""
    
    def __init__(self, threshold_accuracy: float = 0.75, threshold_decline: float = 0.1):
        self.threshold_accuracy = threshold_accuracy
        self.threshold_decline = threshold_decline
        self.performance_history = []
    
    def check_drift(self, current_performance: Dict) -> Dict:
        """Check if model performance has drifted"""
        drift_detected = False
        reasons = []
        
        current_accuracy = current_performance.get('accuracy', 0)
        
        # Check absolute accuracy threshold
        if current_accuracy < self.threshold_accuracy:
            drift_detected = True
            reasons.append(f"Accuracy below threshold: {current_accuracy:.3f} < {self.threshold_accuracy}")
        
        # Check relative decline if we have history
        if len(self.performance_history) > 0:
            recent_avg = np.mean([p['accuracy'] for p in self.performance_history[-5:]])  # Last 5 evaluations
            decline = recent_avg - current_accuracy
            
            if decline > self.threshold_decline:
                drift_detected = True
                reasons.append(f"Performance decline detected: {decline:.3f} from recent average {recent_avg:.3f}")
        
        # Add to history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': current_accuracy,
            'precision': current_performance.get('precision', 0),
            'recall': current_performance.get('recall', 0)
        })
        
        # Keep only last 20 evaluations
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return {
            'drift_detected': drift_detected,
            'reasons': reasons,
            'current_accuracy': current_accuracy,
            'performance_history': self.performance_history[-5:]  # Return last 5 for logging
        }

class ContinuousLearner:
    """Main continuous learning system"""
    
    def __init__(self, fellow_db_path: str, model_dir: str = "models"):
        self.fellow_db_path = Path(fellow_db_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.drift_detector = ModelDriftDetector()
        self.current_model_version = None
        self.last_training_data_count = 0
        
        # Learning configuration
        self.min_training_samples = 10  # Minimum samples needed for retraining
        self.retrain_frequency_days = 7  # Retrain every 7 days
        self.drift_check_frequency_days = 1  # Check for drift daily
        
    def load_fellow_data(self, days_lookback: int = 30) -> pd.DataFrame:
        """Load Fellow call data from database"""
        if not self.fellow_db_path.exists():
            logger.warning(f"Fellow database not found: {self.fellow_db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.fellow_db_path)
            
            # Query recent calls with outcomes
            cutoff_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            
            query = """
            SELECT 
                id, title, company_name, date, ae_name, notes,
                action_items_count, follow_up_scheduled, 
                sentiment_score, strategic_score, processed, enriched
            FROM meetings 
            WHERE date >= ? AND processed = 1
            ORDER BY date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            conn.close()
            
            logger.info(f"Loaded {len(df)} Fellow calls from last {days_lookback} days")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Fellow data: {e}")
            return pd.DataFrame()
    
    def load_enrichment_data(self) -> pd.DataFrame:
        """Load company enrichment data"""
        enrichment_file = Path(__file__).parent.parent.parent / "fellow-enrichment-progress.csv"
        
        if enrichment_file.exists():
            try:
                df = pd.read_csv(enrichment_file)
                logger.info(f"Loaded enrichment data for {len(df)} companies")
                return df
            except Exception as e:
                logger.error(f"Error loading enrichment data: {e}")
        
        return pd.DataFrame()
    
    def create_outcome_labels(self, fellow_data: pd.DataFrame) -> pd.DataFrame:
        """Create ground truth labels from Fellow call outcomes"""
        
        # Extract actual AE progression signals from notes and outcomes
        fellow_data = fellow_data.copy()
        
        # Progression indicators from call outcomes
        progression_signals = []
        voice_ai_signals = []
        
        for idx, row in fellow_data.iterrows():
            notes = str(row.get('notes', '')).lower()
            title = str(row.get('title', '')).lower()
            
            # Strong progression signals
            strong_progression = any(signal in notes for signal in [
                'pricing sent', 'contract sent', 'next meeting scheduled', 
                'technical deep dive', 'demo scheduled', 'poc approved',
                'decision maker meeting', 'implementation discussion'
            ])
            
            # Medium progression signals
            medium_progression = (
                row.get('follow_up_scheduled', 0) == 1 or
                row.get('action_items_count', 0) >= 2 or
                row.get('sentiment_score', 0) >= 8
            )
            
            # Voice AI signals from call content
            voice_ai_mentioned = any(signal in f"{notes} {title}" for signal in [
                'voice ai', 'conversational ai', 'ai calling', 'voice automation',
                'ai voice', 'voice bot', 'ai assistant', 'speech ai'
            ])
            
            # Final labels
            progression_signals.append(1 if strong_progression else (1 if medium_progression else 0))
            voice_ai_signals.append(1 if voice_ai_mentioned else 0)
        
        fellow_data['actual_progression'] = progression_signals
        fellow_data['actual_voice_ai_fit'] = voice_ai_signals
        
        logger.info(f"Created outcome labels - Progression rate: {np.mean(progression_signals):.2%}, "
                   f"Voice AI rate: {np.mean(voice_ai_signals):.2%}")
        
        return fellow_data
    
    def prepare_training_data(self, fellow_data: pd.DataFrame, enrichment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels for training"""
        
        # Create outcome labels
        fellow_data_with_labels = self.create_outcome_labels(fellow_data)
        
        # Map enrichment data to Fellow calls
        company_mapping = {}
        if not enrichment_data.empty:
            for _, row in enrichment_data.iterrows():
                company_name = row.get('Company', '').strip()
                if company_name:
                    company_mapping[company_name] = row.to_dict()
        
        # Create mock company data for unmatched companies
        company_data_list = []
        for company_name in fellow_data_with_labels['company_name'].unique():
            if company_name in company_mapping:
                company_info = company_mapping[company_name]
                company_data_list.append({
                    'company': company_name,
                    'domain': company_info.get('Domain', ''),
                    'industry': company_info.get('Industry', 'Unknown'),
                    'employees': company_info.get('Employees', 'Unknown'),
                    'revenue': company_info.get('Revenue', 'Unknown'),
                    'ai_signals': company_info.get('AI Signals', 'Unknown'),
                    'notes': company_info.get('Notes', '')
                })
            else:
                # Default company data
                company_data_list.append({
                    'company': company_name,
                    'domain': '',
                    'industry': 'Unknown',
                    'employees': 'Unknown', 
                    'revenue': 'Unknown',
                    'ai_signals': 'Unknown',
                    'notes': ''
                })
        
        company_df = pd.DataFrame(company_data_list)
        
        # Extract features
        features_df = self.feature_pipeline.prepare_training_data(fellow_data_with_labels, company_df)
        
        # Create targets
        y_progression = fellow_data_with_labels['actual_progression'].values
        y_voice_ai = fellow_data_with_labels['actual_voice_ai_fit'].values
        
        # Transform features
        X = self.feature_pipeline.fit_transform(features_df)
        feature_names = self.feature_pipeline.feature_columns
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y_progression, y_voice_ai, feature_names
    
    def evaluate_current_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                              model_version: str = None) -> Dict:
        """Evaluate current model performance on new data"""
        
        try:
            # Load current model
            if not model_version and self.current_model_version:
                model_version = self.current_model_version
            
            if not model_version:
                logger.warning("No model version specified for evaluation")
                return {}
            
            version_dir = self.model_dir / f"v_{model_version}"
            if not version_dir.exists():
                logger.warning(f"Model version {model_version} not found")
                return {}
            
            # Find progression model
            progression_model_path = None
            for model_file in version_dir.glob("*progression*.joblib"):
                progression_model_path = model_file
                break
            
            if not progression_model_path:
                logger.warning("Progression model not found")
                return {}
            
            # Load and evaluate model
            model = joblib.load(progression_model_path)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'model_version': model_version,
                'evaluation_date': datetime.now().isoformat(),
                'test_samples': len(y_test)
            }
            
            logger.info(f"Model {model_version} evaluation - Accuracy: {performance['accuracy']:.3f}, "
                       f"Precision: {performance['precision']:.3f}, Recall: {performance['recall']:.3f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def should_retrain(self, current_data_count: int, days_since_last_training: int) -> Tuple[bool, List[str]]:
        """Determine if model should be retrained"""
        
        should_retrain = False
        reasons = []
        
        # Check if enough new data is available
        new_samples = current_data_count - self.last_training_data_count
        if new_samples >= self.min_training_samples:
            should_retrain = True
            reasons.append(f"New training samples available: {new_samples}")
        
        # Check time-based retraining
        if days_since_last_training >= self.retrain_frequency_days:
            should_retrain = True
            reasons.append(f"Scheduled retraining: {days_since_last_training} days since last training")
        
        # Check for minimum data requirement
        if current_data_count < self.min_training_samples:
            should_retrain = False
            reasons = [f"Insufficient training data: {current_data_count} < {self.min_training_samples}"]
        
        return should_retrain, reasons
    
    def retrain_model(self, X: np.ndarray, y_progression: np.ndarray, y_voice_ai: np.ndarray, 
                     feature_names: List[str]) -> str:
        """Retrain models with new data"""
        
        logger.info("Starting model retraining")
        
        # Split data for training and validation
        test_size = min(0.3, max(0.1, 20 / len(X)))  # Adaptive test size based on data amount
        
        X_train, X_val, y_prog_train, y_prog_val = train_test_split(
            X, y_progression, test_size=test_size, random_state=42, stratify=y_progression
        )
        
        _, _, y_va_train, y_va_val = train_test_split(
            X, y_voice_ai, test_size=test_size, random_state=42, stratify=y_voice_ai
        )
        
        # Initialize trainer
        trainer = ModelTrainer(model_dir=str(self.model_dir))
        
        # Train progression models
        progression_results = trainer.train_progression_model(X_train, y_prog_train, feature_names)
        
        # Train Voice AI models
        voice_ai_results = trainer.train_voice_ai_model(X_train, y_va_train, feature_names)
        
        # Evaluate on validation set
        evaluation_results = trainer.evaluate_models(X_val, y_prog_val, y_va_val)
        
        # Save models with timestamp version
        model_version = trainer.save_models(f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Update tracking
        self.current_model_version = model_version
        self.last_training_data_count = len(X)
        
        # Log results
        best_models = trainer.select_best_models()
        if best_models.get('progression'):
            best_perf = evaluation_results[best_models['progression']]
            logger.info(f"Retraining complete - Best model: {best_models['progression']} "
                       f"(Accuracy: {best_perf['accuracy']:.3f})")
        
        return model_version
    
    def run_continuous_learning_cycle(self, force_retrain: bool = False) -> Dict:
        """Run a complete continuous learning cycle"""
        
        logger.info("Starting continuous learning cycle")
        cycle_start = datetime.now()
        
        results = {
            'cycle_start': cycle_start.isoformat(),
            'actions_taken': [],
            'model_version_before': self.current_model_version,
            'model_version_after': self.current_model_version,
            'performance_before': {},
            'performance_after': {},
            'drift_check': {},
            'retrain_decision': {},
            'errors': []
        }
        
        try:
            # Load data
            fellow_data = self.load_fellow_data(days_lookback=60)  # Last 2 months
            enrichment_data = self.load_enrichment_data()
            
            if fellow_data.empty:
                logger.warning("No Fellow data available for continuous learning")
                results['errors'].append("No Fellow data available")
                return results
            
            # Prepare training data
            X, y_progression, y_voice_ai, feature_names = self.prepare_training_data(
                fellow_data, enrichment_data
            )
            
            # Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_progression, test_size=0.2, random_state=42, stratify=y_progression
            )
            
            # Evaluate current model performance if available
            if self.current_model_version:
                current_performance = self.evaluate_current_model(X_test, y_test, self.current_model_version)
                results['performance_before'] = current_performance
                
                # Check for drift
                drift_check = self.drift_detector.check_drift(current_performance)
                results['drift_check'] = drift_check
                
                if drift_check['drift_detected']:
                    logger.warning(f"Model drift detected: {'; '.join(drift_check['reasons'])}")
                    results['actions_taken'].append("model_drift_detected")
            
            # Determine if retraining is needed
            days_since_training = 999  # Default to high value if no previous training
            should_retrain, retrain_reasons = self.should_retrain(len(X), days_since_training)
            
            # Force retrain if drift detected or forced
            if force_retrain or (results.get('drift_check', {}).get('drift_detected', False)):
                should_retrain = True
                if force_retrain:
                    retrain_reasons.append("forced_retrain_requested")
                if results.get('drift_check', {}).get('drift_detected', False):
                    retrain_reasons.append("model_drift_detected")
            
            results['retrain_decision'] = {
                'should_retrain': should_retrain,
                'reasons': retrain_reasons,
                'training_samples': len(X)
            }
            
            # Retrain if needed
            if should_retrain:
                logger.info(f"Retraining triggered: {'; '.join(retrain_reasons)}")
                
                new_model_version = self.retrain_model(X, y_progression, y_voice_ai, feature_names)
                results['model_version_after'] = new_model_version
                results['actions_taken'].append("model_retrained")
                
                # Evaluate new model
                new_performance = self.evaluate_current_model(X_test, y_test, new_model_version)
                results['performance_after'] = new_performance
                
                logger.info(f"Continuous learning cycle complete - New model: {new_model_version}")
            else:
                logger.info(f"No retraining needed: {'; '.join(retrain_reasons) if retrain_reasons else 'No triggers met'}")
            
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {e}")
            results['errors'].append(str(e))
        
        cycle_end = datetime.now()
        results['cycle_end'] = cycle_end.isoformat()
        results['duration_minutes'] = (cycle_end - cycle_start).total_seconds() / 60
        
        return results
    
    def save_cycle_results(self, results: Dict, log_file: str = "continuous_learning_log.json"):
        """Save continuous learning cycle results to log file"""
        
        log_path = self.model_dir / log_file
        
        # Load existing log
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'cycles': []}
        
        # Add new cycle
        log_data['cycles'].append(results)
        
        # Keep only last 50 cycles
        if len(log_data['cycles']) > 50:
            log_data['cycles'] = log_data['cycles'][-50:]
        
        # Save updated log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Cycle results saved to {log_path}")

def run_daily_learning():
    """Run daily continuous learning process"""
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting daily continuous learning process")
    
    # Initialize continuous learner
    fellow_db = Path(__file__).parent.parent / "data" / "fellow_data.db"
    learner = ContinuousLearner(fellow_db, "models")
    
    # Run learning cycle
    results = learner.run_continuous_learning_cycle()
    
    # Save results
    learner.save_cycle_results(results)
    
    # Print summary
    print("\nContinuous Learning Summary:")
    print(f"Duration: {results.get('duration_minutes', 0):.1f} minutes")
    print(f"Actions taken: {', '.join(results.get('actions_taken', ['none']))}")
    print(f"Model version: {results.get('model_version_before', 'none')} → {results.get('model_version_after', 'none')}")
    
    if results.get('performance_after'):
        perf = results['performance_after']
        print(f"New model performance - Accuracy: {perf.get('accuracy', 0):.3f}, "
              f"Precision: {perf.get('precision', 0):.3f}")
    
    if results.get('errors'):
        print(f"Errors: {'; '.join(results['errors'])}")

if __name__ == "__main__":
    run_daily_learning()