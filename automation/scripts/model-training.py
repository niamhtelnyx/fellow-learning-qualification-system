#!/usr/bin/env python3
"""
Model Training Script for Fellow.ai Learning System
Trains and updates the lead qualification model based on Fellow call outcomes
Runs weekly to improve qualification accuracy
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import traceback
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(DATA_DIR, "fellow_data.db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging setup
log_file = os.path.join(LOG_DIR, f"model-training-{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass

class FellowLearningTrainer:
    """Main class for training the Fellow qualification learning model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        self.training_data = []
        self.model_version = f"v{datetime.now().strftime('%Y%m%d')}"
    
    def collect_training_data(self, min_sentiment_threshold: int = 6) -> List[Dict]:
        """Collect training data from Fellow meetings with outcomes"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Get meetings with enrichment data for training
            cursor.execute('''
                SELECT m.id, m.company_name, m.title, m.notes, m.sentiment_score,
                       m.action_items_count, m.follow_up_scheduled, m.date, m.ae_name,
                       e.combined_data, ls.final_score
                FROM meetings m
                LEFT JOIN enrichment_data e ON m.id = e.meeting_id
                LEFT JOIN lead_scores ls ON m.id = ls.meeting_id
                WHERE m.sentiment_score >= ? AND m.notes IS NOT NULL 
                ORDER BY m.created_at DESC
                LIMIT 1000
            ''', (min_sentiment_threshold,))
            
            rows = cursor.fetchall()
            
            training_samples = []
            for row in rows:
                meeting_id, company, title, notes, sentiment, action_items, follow_up, date, ae, enrichment_json, lead_score = row
                
                # Parse enrichment data
                enrichment_data = {}
                if enrichment_json:
                    try:
                        enrichment_data = json.loads(enrichment_json)
                    except:
                        pass
                
                # Create training sample
                sample = {
                    'meeting_id': meeting_id,
                    'company_name': company,
                    'title': title,
                    'notes': notes,
                    'sentiment_score': sentiment,
                    'action_items_count': action_items,
                    'follow_up_scheduled': bool(follow_up),
                    'date': date,
                    'ae_name': ae,
                    'enrichment_data': enrichment_data,
                    'lead_score': lead_score,
                    'target_class': self.determine_target_class(sentiment, enrichment_data, notes)
                }
                
                training_samples.append(sample)
            
            logger.info(f"Collected {len(training_samples)} training samples")
            return training_samples
            
        finally:
            conn.close()
    
    def determine_target_class(self, sentiment_score: int, enrichment_data: Dict, notes: str) -> int:
        """Determine target class for training (0=low value, 1=high value)"""
        # Use multiple signals to determine if this was a high-value lead
        
        # High sentiment score is a strong signal
        if sentiment_score >= 9:
            return 1
        
        # Check for high-value keywords in notes
        high_value_keywords = [
            'millions', '100k+', '50k+', 'enterprise', 'urgent', 'needed months ago',
            'scaling', 'rapid growth', 'immediate', 'ready to move', 'commitment'
        ]
        
        notes_lower = notes.lower() if notes else ''
        if any(keyword in notes_lower for keyword in high_value_keywords):
            return 1
        
        # Check enrichment signals
        if enrichment_data:
            clearbit = enrichment_data.get('clearbit', {})
            ai_signals = enrichment_data.get('ai_signals', {})
            
            # Large company size
            employees = clearbit.get('employees', 0)
            if isinstance(employees, str) and '-' in employees:
                try:
                    employees = int(employees.split('-')[1])
                except:
                    employees = 0
            
            if employees >= 500:
                return 1
            
            # High AI signal score
            if ai_signals.get('score', 0) >= 50:
                return 1
        
        # Medium sentiment with positive signals
        if sentiment_score >= 7:
            if any(keyword in notes_lower for keyword in [
                'integration', 'api', 'technical', 'developers', 'platform'
            ]):
                return 1
        
        # Default to low value
        return 0
    
    def extract_features(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and target labels from training data"""
        numerical_features = []
        text_features = []
        targets = []
        
        for sample in training_data:
            # Extract numerical features
            features = {
                'sentiment_score': sample.get('sentiment_score', 5),
                'action_items_count': sample.get('action_items_count', 0),
                'has_follow_up': int(sample.get('follow_up_scheduled', False)),
                'notes_length': len(sample.get('notes', ''))
            }
            
            # Enrichment-based features
            enrichment = sample.get('enrichment_data', {})
            if enrichment:
                clearbit = enrichment.get('clearbit', {})
                ai_signals = enrichment.get('ai_signals', {})
                
                # Company size
                employees = clearbit.get('employees', 0)
                if isinstance(employees, str) and '-' in employees:
                    try:
                        employees = int(employees.split('-')[1])
                    except:
                        employees = 0
                
                features.update({
                    'company_size': employees,
                    'company_size_log': np.log1p(employees),
                    'voice_ai_primary': int(ai_signals.get('voice_ai_primary', False)),
                    'ai_signals_score': ai_signals.get('score', 0),
                    'uses_modern_tech': int(len(clearbit.get('technologies', [])) > 0),
                    'tech_count': len(clearbit.get('technologies', []))
                })
            else:
                features.update({
                    'company_size': 0,
                    'company_size_log': 0,
                    'voice_ai_primary': 0,
                    'ai_signals_score': 0,
                    'uses_modern_tech': 0,
                    'tech_count': 0
                })
            
            numerical_features.append(features)
            
            # Text features
            text_content = ' '.join([
                sample.get('company_name', ''),
                sample.get('notes', ''),
                sample.get('title', '')
            ])
            text_features.append(text_content)
            
            # Target
            targets.append(sample.get('target_class', 0))
        
        # Convert to arrays
        numerical_df = pd.DataFrame(numerical_features)
        numerical_array = numerical_df.values
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        text_array = self.vectorizer.fit_transform(text_features).toarray()
        
        # Combine features
        combined_features = np.hstack([numerical_array, text_array])
        
        # Create feature names
        self.feature_names = list(numerical_df.columns) + [f'text_{i}' for i in range(text_array.shape[1])]
        
        targets_array = np.array(targets)
        
        logger.info(f"Extracted features: {combined_features.shape[1]} features, {len(targets)} samples")
        logger.info(f"Class distribution: {np.bincount(targets_array)}")
        
        return combined_features, targets_array
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the qualification model"""
        
        # Check if we have enough data
        if len(X) < 50:
            raise ModelTrainingError(f"Insufficient training data: {len(X)} samples (need at least 50)")
        
        # Check class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            raise ModelTrainingError("Need both high-value and low-value samples for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        # Feature importance
        feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        results = {
            'model_version': self.model_version,
            'training_samples': len(X),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_f1_mean': float(np.mean(cv_scores)),
            'cv_f1_std': float(np.std(cv_scores)),
            'class_distribution': {str(cls): int(count) for cls, count in zip(unique_classes, class_counts)},
            'feature_importance_top10': feature_importance[:10],
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Model training complete!")
        logger.info(f"Test accuracy: {test_score:.3f}")
        logger.info(f"Cross-validation F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        return results
    
    def save_model(self):
        """Save the trained model to disk"""
        # Save model
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'version': self.model_version,
            'created_at': datetime.now().isoformat()
        }
        
        model_path = os.path.join(MODELS_DIR, "qualification_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save backup with version
        backup_model_path = os.path.join(MODELS_DIR, f"qualification_model_{self.model_version}.pkl")
        with open(backup_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        backup_vectorizer_path = os.path.join(MODELS_DIR, f"text_vectorizer_{self.model_version}.pkl")
        with open(backup_vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Backup saved: {backup_model_path}")
    
    def log_training_results(self, results: Dict):
        """Log training results to database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create model_training_log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    training_samples INTEGER,
                    test_accuracy REAL,
                    cv_f1_mean REAL,
                    cv_f1_std REAL,
                    results_json TEXT
                )
            ''')
            
            # Store results
            cursor.execute('''
                INSERT INTO model_training_log (
                    model_version, training_samples, test_accuracy, 
                    cv_f1_mean, cv_f1_std, results_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                results['model_version'],
                results['training_samples'],
                results['test_accuracy'],
                results['cv_f1_mean'],
                results['cv_f1_std'],
                json.dumps(results)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log training results: {e}")
    
    def run_training_pipeline(self) -> Dict:
        """Run the complete training pipeline"""
        logger.info("Starting Fellow.ai model training pipeline")
        
        try:
            # Collect training data
            training_data = self.collect_training_data()
            
            if len(training_data) < 50:
                raise ModelTrainingError(f"Insufficient training data: {len(training_data)} samples")
            
            # Extract features
            X, y = self.extract_features(training_data)
            
            # Train model
            results = self.train_model(X, y)
            
            # Save model
            self.save_model()
            
            # Log results
            self.log_training_results(results)
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main entry point for model training"""
    try:
        trainer = FellowLearningTrainer()
        results = trainer.run_training_pipeline()
        
        # Output results
        print(json.dumps(results, indent=2))
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()