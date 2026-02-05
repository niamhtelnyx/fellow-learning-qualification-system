#!/usr/bin/env python3
"""
Real-time Lead Scoring System
Processes new leads through qualification model and scores them
Runs hourly during business hours to catch new opportunities
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
log_file = os.path.join(LOG_DIR, f"realtime-scoring-{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScoringError(Exception):
    """Custom exception for scoring errors"""
    pass

class LeadQualificationModel:
    """Machine learning model for lead qualification scoring"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        self.model_version = "v1.0"
        self.loaded = False
    
    def load_model(self) -> bool:
        """Load the trained model from disk"""
        try:
            model_path = os.path.join(MODELS_DIR, "qualification_model.pkl")
            vectorizer_path = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data['feature_names']
                    self.model_version = model_data.get('version', 'v1.0')
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.loaded = True
                logger.info(f"Loaded qualification model {self.model_version}")
                return True
            else:
                logger.warning("No trained model found, using fallback scoring")
                self.loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.loaded = False
            return False
    
    def extract_numerical_features(self, meeting_data: dict, enrichment_data: dict) -> dict:
        """Extract numerical features from meeting and enrichment data"""
        features = {}
        
        # Meeting-based features
        features['sentiment_score'] = meeting_data.get('sentiment_score', 5)
        features['action_items_count'] = meeting_data.get('action_items_count', 0)
        features['has_follow_up'] = int(meeting_data.get('follow_up_scheduled', False))
        features['notes_length'] = len(meeting_data.get('notes', ''))
        
        # Enrichment-based features
        if enrichment_data:
            clearbit = enrichment_data.get('clearbit', {})
            ai_signals = enrichment_data.get('ai_signals', {})
            
            # Company size features
            employees = clearbit.get('employees', 0)
            if isinstance(employees, str):
                # Parse employee range (e.g., "50-100")
                if '-' in employees:
                    try:
                        employees = int(employees.split('-')[1])
                    except:
                        employees = 0
                else:
                    try:
                        employees = int(employees)
                    except:
                        employees = 0
            
            features['company_size'] = employees
            features['company_size_score'] = self.score_company_size(employees)
            
            # Revenue features
            revenue = clearbit.get('annual_revenue', 0)
            if isinstance(revenue, str):
                revenue = self.parse_revenue_string(revenue)
            features['annual_revenue'] = revenue
            features['revenue_score'] = self.score_revenue(revenue)
            
            # AI signals
            features['voice_ai_primary'] = int(ai_signals.get('voice_ai_primary', False))
            features['ai_technology_user'] = int(ai_signals.get('ai_technology_user', False))
            features['automation_focus'] = int(ai_signals.get('automation_focus', False))
            features['ai_signals_score'] = ai_signals.get('score', 0)
            
            # Technology features
            technologies = clearbit.get('technologies', [])
            features['uses_modern_tech'] = int(any(
                tech.lower() in ['react', 'node.js', 'python', 'aws', 'api'] 
                for tech in technologies
            ))
            features['tech_count'] = len(technologies)
            
            # Industry features
            industry = clearbit.get('industry', '')
            features['high_value_industry'] = int(industry.lower() in [
                'software', 'technology', 'saas', 'telecommunications', 
                'healthcare', 'financial services', 'e-commerce'
            ])
        else:
            # Default values when no enrichment data
            features.update({
                'company_size': 0,
                'company_size_score': 1,
                'annual_revenue': 0,
                'revenue_score': 1,
                'voice_ai_primary': 0,
                'ai_technology_user': 0,
                'automation_focus': 0,
                'ai_signals_score': 0,
                'uses_modern_tech': 0,
                'tech_count': 0,
                'high_value_industry': 0
            })
        
        return features
    
    def extract_text_features(self, meeting_data: dict) -> str:
        """Extract and combine text features for vectorization"""
        text_parts = []
        
        # Company name
        company_name = meeting_data.get('company_name', '')
        if company_name and company_name != 'Unknown Company':
            text_parts.append(company_name)
        
        # Meeting notes
        notes = meeting_data.get('notes', '')
        if notes:
            text_parts.append(notes)
        
        # Meeting title
        title = meeting_data.get('title', '')
        if title:
            text_parts.append(title)
        
        return ' '.join(text_parts)
    
    def score_company_size(self, employees: int) -> int:
        """Score company based on employee count"""
        if employees >= 1000:
            return 5
        elif employees >= 500:
            return 4
        elif employees >= 100:
            return 3
        elif employees >= 50:
            return 2
        elif employees >= 10:
            return 1
        else:
            return 0
    
    def score_revenue(self, revenue: int) -> int:
        """Score company based on annual revenue"""
        if revenue >= 100_000_000:  # $100M+
            return 5
        elif revenue >= 50_000_000:  # $50M+
            return 4
        elif revenue >= 10_000_000:  # $10M+
            return 3
        elif revenue >= 1_000_000:   # $1M+
            return 2
        elif revenue >= 100_000:     # $100K+
            return 1
        else:
            return 0
    
    def parse_revenue_string(self, revenue_str: str) -> int:
        """Parse revenue string to integer"""
        if not revenue_str:
            return 0
        
        revenue_str = revenue_str.lower().replace(',', '').replace('$', '')
        
        # Handle common formats
        if 'million' in revenue_str or 'm' in revenue_str:
            try:
                number = float(revenue_str.replace('million', '').replace('m', '').strip())
                return int(number * 1_000_000)
            except:
                return 0
        elif 'billion' in revenue_str or 'b' in revenue_str:
            try:
                number = float(revenue_str.replace('billion', '').replace('b', '').strip())
                return int(number * 1_000_000_000)
            except:
                return 0
        else:
            try:
                return int(float(revenue_str))
            except:
                return 0
    
    def predict_qualification_score(self, meeting_data: dict, enrichment_data: dict) -> Dict:
        """Predict qualification score for a lead"""
        try:
            # Extract features
            numerical_features = self.extract_numerical_features(meeting_data, enrichment_data)
            text_content = self.extract_text_features(meeting_data)
            
            if self.loaded and self.model and self.vectorizer:
                # Use ML model prediction
                
                # Vectorize text
                text_vector = self.vectorizer.transform([text_content]).toarray()[0]
                
                # Combine numerical and text features
                feature_vector = []
                for feature_name in self.feature_names:
                    if feature_name.startswith('text_'):
                        # Text feature
                        idx = int(feature_name.split('_')[1])
                        if idx < len(text_vector):
                            feature_vector.append(text_vector[idx])
                        else:
                            feature_vector.append(0)
                    else:
                        # Numerical feature
                        feature_vector.append(numerical_features.get(feature_name, 0))
                
                # Predict probability
                X = np.array(feature_vector).reshape(1, -1)
                prediction_proba = self.model.predict_proba(X)[0]
                
                # Get prediction score (probability of high-value lead)
                high_value_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                ml_score = int(high_value_prob * 100)
                
                return {
                    'method': 'ml_model',
                    'model_version': self.model_version,
                    'ml_score': ml_score,
                    'fallback_score': self.calculate_fallback_score(numerical_features, text_content),
                    'final_score': ml_score,
                    'features_used': numerical_features,
                    'confidence': float(max(prediction_proba))
                }
            else:
                # Use fallback rule-based scoring
                fallback_score = self.calculate_fallback_score(numerical_features, text_content)
                
                return {
                    'method': 'fallback_rules',
                    'model_version': 'fallback_v1',
                    'ml_score': None,
                    'fallback_score': fallback_score,
                    'final_score': fallback_score,
                    'features_used': numerical_features,
                    'confidence': 0.7  # Lower confidence for rule-based
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            # Emergency fallback
            emergency_score = meeting_data.get('sentiment_score', 5) * 10
            return {
                'method': 'emergency_fallback',
                'model_version': 'emergency_v1',
                'ml_score': None,
                'fallback_score': emergency_score,
                'final_score': emergency_score,
                'features_used': {},
                'confidence': 0.5,
                'error': str(e)
            }
    
    def calculate_fallback_score(self, features: dict, text_content: str) -> int:
        """Calculate score using rule-based approach"""
        score = 50  # Base score
        
        # Sentiment score influence (30% weight)
        sentiment_component = (features.get('sentiment_score', 5) - 5) * 6
        score += sentiment_component
        
        # Company size influence (20% weight)
        size_component = features.get('company_size_score', 0) * 4
        score += size_component
        
        # Revenue influence (20% weight)
        revenue_component = features.get('revenue_score', 0) * 4
        score += revenue_component
        
        # AI signals influence (20% weight)
        ai_component = features.get('ai_signals_score', 0) / 5 * 20
        score += ai_component
        
        # Engagement influence (10% weight)
        engagement_score = 0
        if features.get('action_items_count', 0) > 2:
            engagement_score += 5
        if features.get('has_follow_up', 0):
            engagement_score += 5
        score += engagement_score
        
        # Text-based bonuses
        text_lower = text_content.lower()
        
        # High-value keywords
        if any(keyword in text_lower for keyword in [
            'millions', '100k+', '50k+', 'enterprise', 'scale', 'urgent'
        ]):
            score += 10
        
        # Technology keywords
        if any(keyword in text_lower for keyword in [
            'api', 'integration', 'platform', 'developer', 'technical'
        ]):
            score += 5
        
        # Risk factors
        if any(keyword in text_lower for keyword in [
            'budget', 'evaluation', 'comparing', 'thinking about'
        ]):
            score -= 5
        
        return max(0, min(100, int(score)))

class RealtimeScorer:
    """Main class for real-time lead scoring"""
    
    def __init__(self):
        self.model = LeadQualificationModel()
        self.model.load_model()
    
    def get_unscored_meetings(self, hours_back: int = 24) -> List[Dict]:
        """Get meetings that need scoring"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Get meetings from last N hours that haven't been scored yet
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            cursor.execute('''
                SELECT m.id, m.title, m.company_name, m.date, m.ae_name, 
                       m.notes, m.action_items_count, m.follow_up_scheduled,
                       m.sentiment_score, m.created_at, e.combined_data
                FROM meetings m
                LEFT JOIN enrichment_data e ON m.id = e.meeting_id
                LEFT JOIN lead_scores ls ON m.id = ls.meeting_id
                WHERE m.created_at >= ? AND ls.meeting_id IS NULL
                ORDER BY m.created_at DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            
            meetings = []
            for row in rows:
                enrichment_data = {}
                if row[10]:  # combined_data column
                    try:
                        enrichment_data = json.loads(row[10])
                    except:
                        pass
                
                meeting = {
                    'id': row[0],
                    'title': row[1],
                    'company_name': row[2],
                    'date': row[3],
                    'ae_name': row[4],
                    'notes': row[5],
                    'action_items_count': row[6],
                    'follow_up_scheduled': bool(row[7]),
                    'sentiment_score': row[8],
                    'created_at': row[9],
                    'enrichment_data': enrichment_data
                }
                meetings.append(meeting)
            
            return meetings
            
        finally:
            conn.close()
    
    def store_lead_score(self, meeting_id: str, score_data: Dict):
        """Store lead qualification score"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Create lead_scores table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lead_scores (
                    meeting_id TEXT PRIMARY KEY,
                    final_score INTEGER,
                    method TEXT,
                    model_version TEXT,
                    ml_score INTEGER,
                    fallback_score INTEGER,
                    confidence REAL,
                    features_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (meeting_id) REFERENCES meetings (id)
                )
            ''')
            
            # Store score
            cursor.execute('''
                INSERT OR REPLACE INTO lead_scores (
                    meeting_id, final_score, method, model_version,
                    ml_score, fallback_score, confidence, features_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meeting_id,
                score_data['final_score'],
                score_data['method'],
                score_data['model_version'],
                score_data.get('ml_score'),
                score_data.get('fallback_score'),
                score_data['confidence'],
                json.dumps(score_data.get('features_used', {}))
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def score_leads(self) -> Dict:
        """Score all unscored leads"""
        logger.info("Starting real-time lead scoring")
        
        results = {
            'processed': 0,
            'scored': 0,
            'high_value': 0,
            'errors': 0,
            'error_details': []
        }
        
        # Get unscored meetings
        meetings = self.get_unscored_meetings()
        
        if not meetings:
            logger.info("No meetings found requiring scoring")
            return results
        
        logger.info(f"Found {len(meetings)} meetings to score")
        
        for meeting in meetings:
            try:
                # Score the meeting
                score_data = self.model.predict_qualification_score(
                    meeting, 
                    meeting['enrichment_data']
                )
                
                # Store the score
                self.store_lead_score(meeting['id'], score_data)
                
                results['scored'] += 1
                
                # Check if high value
                if score_data['final_score'] >= 80:
                    results['high_value'] += 1
                    logger.info(f"HIGH VALUE LEAD: {meeting['company_name']} - Score: {score_data['final_score']}")
                
                logger.info(f"Scored {meeting['company_name']}: {score_data['final_score']} ({score_data['method']})")
                
            except Exception as e:
                error_msg = f"Scoring failed for {meeting['company_name']}: {str(e)}"
                logger.error(error_msg)
                results['errors'] += 1
                results['error_details'].append(error_msg)
            
            results['processed'] += 1
        
        logger.info(f"Scoring complete: {results['scored']} scored, {results['high_value']} high-value leads")
        return results
    
    def get_recent_high_value_leads(self, hours_back: int = 24, min_score: int = 80) -> List[Dict]:
        """Get recent high-value leads for alerting"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            cursor.execute('''
                SELECT m.company_name, m.ae_name, m.date, ls.final_score, 
                       ls.method, ls.confidence, m.notes
                FROM meetings m
                JOIN lead_scores ls ON m.id = ls.meeting_id
                WHERE ls.created_at >= ? AND ls.final_score >= ?
                ORDER BY ls.final_score DESC, ls.created_at DESC
            ''', (cutoff_time, min_score))
            
            rows = cursor.fetchall()
            
            leads = []
            for row in rows:
                leads.append({
                    'company_name': row[0],
                    'ae_name': row[1],
                    'date': row[2],
                    'score': row[3],
                    'method': row[4],
                    'confidence': row[5],
                    'notes_preview': row[6][:200] + '...' if len(row[6]) > 200 else row[6]
                })
            
            return leads
            
        finally:
            conn.close()

def main():
    """Main entry point for real-time scoring"""
    try:
        scorer = RealtimeScorer()
        
        # Score unscored leads
        results = scorer.score_leads()
        
        # Get high-value leads for potential alerting
        high_value_leads = scorer.get_recent_high_value_leads()
        
        # Combine results
        output = {
            'scoring_results': results,
            'high_value_leads': high_value_leads,
            'timestamp': datetime.now().isoformat()
        }
        
        # Output results
        print(json.dumps(output, indent=2))
        
        # Exit with error if scoring failed completely
        if results['processed'] > 0 and results['scored'] == 0:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Real-time scoring failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()