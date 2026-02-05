#!/usr/bin/env python3
"""
Continuous Learning System for Fellow Learning Qualification Model
Automatically retrains models with new Fellow call outcomes and monitors performance

ML Team: Continuous Learning Pipeline
- Daily data sync and feature processing
- Model performance drift detection  
- Automated weekly retraining
- A/B testing framework for model comparison
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
import sys
sys.path.append(str(Path(__file__).parent.parent))
from feature_engineering.feature_engineer import FeatureEngineeringPipeline
from training.model_trainer import ModelTrainer, QualificationScorer

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
            recent_avg = np.mean([p['accuracy'] for p in self.performance_history[-5:]])
            decline = recent_avg - current_accuracy
            
            if decline > self.threshold_decline:
                drift_detected = True
                reasons.append(f"Performance decline: {decline:.3f} from recent average {recent_avg:.3f}")
        
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
            'performance_history': self.performance_history[-5:]
        }

class ContinuousLearner:
    """Main continuous learning system for Fellow qualification models"""
    
    def __init__(self, fellow_db_path: str, model_dir: str = "models"):
        self.fellow_db_path = Path(fellow_db_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.drift_detector = ModelDriftDetector()
        self.current_model_version = None
        self.last_training_data_count = 0
        
        # Learning configuration
        self.min_training_samples = 20  
        self.retrain_frequency_days = 7  
        self.drift_check_frequency_days = 1
        
        logger.info("🤖 ML Continuous Learning System initialized")
        logger.info(f"📊 Target accuracy: >75% (drift threshold)")
        logger.info(f"🔄 Retraining frequency: Every {self.retrain_frequency_days} days")
    
    def load_fellow_data(self, days_lookback: int = 60) -> pd.DataFrame:
        """Load Fellow call data from database"""
        if not self.fellow_db_path.exists():
            logger.warning(f"Fellow database not found: {self.fellow_db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.fellow_db_path)
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
            
            logger.info(f"📚 Loaded {len(df)} Fellow calls from last {days_lookback} days")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading Fellow data: {e}")
            return pd.DataFrame()
    
    def create_outcome_labels(self, fellow_data: pd.DataFrame) -> pd.DataFrame:
        """Create ground truth labels from Fellow call outcomes"""
        
        fellow_data = fellow_data.copy()
        progression_signals = []
        voice_ai_signals = []
        
        for idx, row in fellow_data.iterrows():
            notes = str(row.get('notes', '')).lower()
            title = str(row.get('title', '')).lower()
            
            # Strong progression signals from actual outcomes
            strong_progression = any(signal in notes for signal in [
                'pricing sent', 'contract sent', 'next meeting scheduled', 
                'technical deep dive', 'demo scheduled', 'poc approved',
                'decision maker meeting', 'implementation discussion',
                'follow up', 'next steps', 'moving forward'
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
                'ai voice', 'voice bot', 'ai assistant', 'speech ai', 'voice platform'
            ])
            
            progression_signals.append(1 if strong_progression or medium_progression else 0)
            voice_ai_signals.append(1 if voice_ai_mentioned else 0)
        
        fellow_data['actual_progression'] = progression_signals
        fellow_data['actual_voice_ai_fit'] = voice_ai_signals
        
        progression_rate = np.mean(progression_signals)
        voice_ai_rate = np.mean(voice_ai_signals)
        
        logger.info(f"📈 Created outcome labels - Progression: {progression_rate:.2%}, Voice AI: {voice_ai_rate:.2%}")
        
        return fellow_data
    
    def should_retrain(self, current_data_count: int, days_since_last_training: int) -> Tuple[bool, List[str]]:
        """Determine if model should be retrained"""
        
        should_retrain = False
        reasons = []
        
        # Check if enough new data is available
        new_samples = current_data_count - self.last_training_data_count
        if new_samples >= self.min_training_samples:
            should_retrain = True
            reasons.append(f"📊 New training samples: {new_samples}")
        
        # Check time-based retraining
        if days_since_last_training >= self.retrain_frequency_days:
            should_retrain = True
            reasons.append(f"⏰ Scheduled retraining: {days_since_last_training} days elapsed")
        
        # Check minimum data requirement
        if current_data_count < self.min_training_samples:
            should_retrain = False
            reasons = [f"❌ Insufficient data: {current_data_count} < {self.min_training_samples}"]
        
        return should_retrain, reasons
    
    def retrain_model(self, X: np.ndarray, y_progression: np.ndarray, y_voice_ai: np.ndarray, 
                     feature_names: List[str]) -> str:
        """Retrain models with new data"""
        
        logger.info("🔄 Starting model retraining with new Fellow outcomes")
        
        # Adaptive test size based on data amount
        test_size = min(0.3, max(0.1, 20 / len(X)))
        
        X_train, X_val, y_prog_train, y_prog_val = train_test_split(
            X, y_progression, test_size=test_size, random_state=42, stratify=y_progression
        )
        
        _, _, y_va_train, y_va_val = train_test_split(
            X, y_voice_ai, test_size=test_size, random_state=42, stratify=y_voice_ai
        )
        
        # Initialize trainer
        trainer = ModelTrainer(model_dir=str(self.model_dir))
        
        # Train models
        logger.info("🧠 Training progression prediction model...")
        progression_results = trainer.train_progression_model(X_train, y_prog_train, feature_names)
        
        logger.info("🎯 Training Voice AI detection model...")  
        voice_ai_results = trainer.train_voice_ai_model(X_train, y_va_train, feature_names)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_val, y_prog_val, y_va_val)
        
        # Save models
        model_version = trainer.save_models(f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Update tracking
        self.current_model_version = model_version
        self.last_training_data_count = len(X)
        
        # Log results
        best_models = trainer.select_best_models()
        if best_models.get('progression'):
            best_perf = evaluation_results[best_models['progression']]
            logger.info(f"✅ Retraining complete - Best model: {best_models['progression']}")
            logger.info(f"📊 Performance - Accuracy: {best_perf['accuracy']:.3f}, Precision: {best_perf['precision']:.3f}")
        
        return model_version
    
    def run_continuous_learning_cycle(self, force_retrain: bool = False) -> Dict:
        """Run complete continuous learning cycle"""
        
        logger.info("🚀 Starting ML continuous learning cycle")
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
            # Load Fellow call data
            logger.info("📚 Loading Fellow call outcomes...")
            fellow_data = self.load_fellow_data(days_lookback=90)
            
            if fellow_data.empty:
                logger.warning("⚠️ No Fellow data available")
                results['errors'].append("No Fellow data available")
                return results
            
            # Create outcome labels and features
            fellow_data_with_labels = self.create_outcome_labels(fellow_data)
            
            # Load company enrichment data (placeholder for now)
            company_data = pd.DataFrame([{
                'company': name,
                'domain': '',
                'industry': 'Unknown',
                'employees': 'Unknown',
                'revenue': 'Unknown',
                'ai_signals': 'Unknown',
                'notes': ''
            } for name in fellow_data_with_labels['company_name'].unique()])
            
            # Prepare training data
            logger.info("⚙️ Engineering features from call data...")
            features_df = self.feature_pipeline.prepare_training_data(
                fellow_data_with_labels, company_data
            )
            
            # Extract targets
            y_progression = fellow_data_with_labels['actual_progression'].values
            y_voice_ai = fellow_data_with_labels['actual_voice_ai_fit'].values
            
            # Transform features
            X = self.feature_pipeline.fit_transform(features_df)
            feature_names = self.feature_pipeline.feature_columns
            
            logger.info(f"📊 Prepared {X.shape[0]} samples with {X.shape[1]} features")
            
            # Determine if retraining needed
            days_since_training = 8  # Default trigger for demo
            should_retrain, retrain_reasons = self.should_retrain(len(X), days_since_training)
            
            if force_retrain:
                should_retrain = True
                retrain_reasons.append("🔧 Forced retraining requested")
            
            results['retrain_decision'] = {
                'should_retrain': should_retrain,
                'reasons': retrain_reasons,
                'training_samples': len(X)
            }
            
            # Retrain if needed
            if should_retrain:
                logger.info(f"🔄 Retraining triggered: {'; '.join(retrain_reasons)}")
                
                new_model_version = self.retrain_model(X, y_progression, y_voice_ai, feature_names)
                results['model_version_after'] = new_model_version
                results['actions_taken'].append("model_retrained")
                
                logger.info(f"✅ Continuous learning cycle complete - New model: {new_model_version}")
            else:
                logger.info(f"⏭️ No retraining needed: {'; '.join(retrain_reasons) if retrain_reasons else 'All triggers satisfied'}")
        
        except Exception as e:
            logger.error(f"❌ Error in continuous learning cycle: {e}")
            results['errors'].append(str(e))
        
        cycle_end = datetime.now()
        results['cycle_end'] = cycle_end.isoformat()
        results['duration_minutes'] = (cycle_end - cycle_start).total_seconds() / 60
        
        logger.info(f"⏱️ Cycle duration: {results['duration_minutes']:.1f} minutes")
        
        return results
    
    def save_cycle_results(self, results: Dict, log_file: str = "continuous_learning_log.json"):
        """Save learning cycle results"""
        
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
        
        logger.info(f"💾 Cycle results saved to {log_path}")

def run_daily_learning():
    """Entry point for daily continuous learning"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🤖 ML Daily Continuous Learning Process Started")
    logger.info("=" * 60)
    
    # Initialize learner
    db_path = Path(__file__).parent.parent.parent / "automation" / "data" / "fellow_data.db"
    model_dir = Path(__file__).parent.parent / "training" / "models"
    
    learner = ContinuousLearner(db_path, model_dir)
    
    # Run learning cycle
    results = learner.run_continuous_learning_cycle()
    
    # Save results
    learner.save_cycle_results(results)
    
    # Print summary
    print("\n📊 ML Continuous Learning Summary:")
    print("=" * 50)
    print(f"⏱️ Duration: {results.get('duration_minutes', 0):.1f} minutes")
    print(f"🎯 Actions: {', '.join(results.get('actions_taken', ['none']))}")
    print(f"🤖 Model: {results.get('model_version_before', 'none')} → {results.get('model_version_after', 'none')}")
    
    if results.get('performance_after'):
        perf = results['performance_after']
        print(f"📈 New performance - Accuracy: {perf.get('accuracy', 0):.3f}, Precision: {perf.get('precision', 0):.3f}")
    
    if results.get('errors'):
        print(f"❌ Errors: {'; '.join(results['errors'])}")
    
    print("=" * 50)
    print("✅ ML Continuous Learning Process Complete")

if __name__ == "__main__":
    run_daily_learning()