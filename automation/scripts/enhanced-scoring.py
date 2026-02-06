#!/usr/bin/env python3
"""
Enhanced Lead Scoring System with Production Logging
Processes leads through qualification model with comprehensive logging
Integrates enrichment, ML scoring, routing decisions, and outcome tracking
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
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logging_system import QualificationLogger, QualificationMetrics

# Configuration
CLEARBIT_API_KEY = os.environ.get('CLEARBIT_API_KEY', '')
OPENFUNNEL_API_KEY = os.environ.get('OPENFUNNEL_API_KEY', '')

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
log_file = os.path.join(LOG_DIR, f"enhanced-scoring-{datetime.now().strftime('%Y-%m-%d')}.log")
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

class EnrichmentEngine:
    """Handle data enrichment from various providers"""
    
    def __init__(self):
        self.clearbit_api_key = CLEARBIT_API_KEY
        self.openfunnel_api_key = OPENFUNNEL_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Telnyx-Qualification-Engine/2.0'
        })
    
    def enrich_company_clearbit(self, company_name: str, domain: str = None) -> Dict:
        """Enrich company data using Clearbit"""
        start_time = time.time()
        
        request_data = {
            'provider': 'clearbit',
            'company_name': company_name,
            'domain': domain,
            'enrichment_type': 'company_data'
        }
        
        if not self.clearbit_api_key:
            return {
                'success': False,
                'error_message': 'Clearbit API key not configured',
                'response_time_ms': 0,
                'confidence_score': 0,
                'cost_cents': 0
            }
        
        try:
            # Try domain-based lookup first
            if domain:
                url = f"https://company.clearbit.com/v2/companies/find?domain={domain}"
            else:
                # Fallback to name-based search
                url = f"https://company.clearbit.com/v2/companies/find?name={company_name}"
            
            headers = {'Authorization': f'Bearer {self.clearbit_api_key}'}
            response = self.session.get(url, headers=headers, timeout=10)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant fields
                enriched_data = {
                    'company_name': data.get('name', company_name),
                    'domain': data.get('domain'),
                    'industry': data.get('category', {}).get('industry'),
                    'employees': data.get('metrics', {}).get('employees'),
                    'annual_revenue': data.get('metrics', {}).get('annualRevenue'),
                    'technologies': data.get('tech', []),
                    'description': data.get('description'),
                    'founded': data.get('foundedYear'),
                    'location': data.get('geo', {}).get('city')
                }
                
                return {
                    'success': True,
                    'data': enriched_data,
                    'response_time_ms': response_time_ms,
                    'confidence_score': 0.85,  # Clearbit is generally reliable
                    'cost_cents': 5  # Approximate cost
                }
            
            elif response.status_code == 404:
                return {
                    'success': False,
                    'error_message': 'Company not found in Clearbit',
                    'response_time_ms': response_time_ms,
                    'confidence_score': 0,
                    'cost_cents': 1  # Still costs for the lookup
                }
            
            else:
                return {
                    'success': False,
                    'error_message': f'Clearbit API error: {response.status_code}',
                    'response_time_ms': response_time_ms,
                    'confidence_score': 0,
                    'cost_cents': 1
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': f'Clearbit enrichment failed: {str(e)}',
                'response_time_ms': int((time.time() - start_time) * 1000),
                'confidence_score': 0,
                'cost_cents': 0
            }
    
    def detect_ai_signals(self, company_name: str, company_data: Dict, notes: str) -> Dict:
        """Detect AI/Voice AI signals from company data and notes"""
        start_time = time.time()
        
        ai_score = 0
        voice_ai_score = 0
        signals_found = []
        
        # Check company name for AI indicators
        ai_keywords_company = [
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'voice', 'speech', 'conversation', 'chatbot', 'bot',
            'automation', 'smart', 'intelligent'
        ]
        
        company_lower = company_name.lower()
        for keyword in ai_keywords_company:
            if keyword in company_lower:
                ai_score += 10
                signals_found.append(f"company_name:{keyword}")
        
        # Check company description and industry
        description = company_data.get('description', '') or ''
        industry = company_data.get('industry', '') or ''
        
        combined_text = f"{description} {industry}".lower()
        
        ai_indicators = [
            ('artificial intelligence', 15), ('machine learning', 15), ('voice ai', 20),
            ('speech recognition', 15), ('natural language', 12), ('chatbot', 10),
            ('conversational ai', 18), ('voice assistant', 18), ('speech technology', 15),
            ('automation', 8), ('intelligent', 5), ('smart', 5)
        ]
        
        for indicator, points in ai_indicators:
            if indicator in combined_text:
                ai_score += points
                if 'voice' in indicator or 'speech' in indicator or 'conversational' in indicator:
                    voice_ai_score += points
                signals_found.append(f"description:{indicator}")
        
        # Check technologies used
        technologies = company_data.get('technologies', [])
        ai_tech_indicators = [
            'tensorflow', 'pytorch', 'keras', 'openai', 'azure cognitive',
            'google cloud ai', 'amazon lex', 'dialogflow', 'rasa',
            'speech-to-text', 'text-to-speech', 'nlp', 'nlu'
        ]
        
        for tech in technologies:
            tech_lower = tech.lower()
            for indicator in ai_tech_indicators:
                if indicator in tech_lower:
                    ai_score += 12
                    if 'speech' in indicator or 'voice' in indicator:
                        voice_ai_score += 12
                    signals_found.append(f"technology:{indicator}")
        
        # Analyze meeting notes for AI discussion
        notes_lower = notes.lower()
        notes_ai_indicators = [
            ('voice ai', 25), ('speech recognition', 20), ('chatbot', 15),
            ('artificial intelligence', 15), ('machine learning', 15),
            ('automated calls', 20), ('voice assistant', 20), ('conversational', 15),
            ('ai integration', 18), ('voice technology', 18), ('speech technology', 18),
            ('voice solution', 16), ('ai solution', 16), ('automation', 10)
        ]
        
        for indicator, points in notes_ai_indicators:
            if indicator in notes_lower:
                ai_score += points
                if 'voice' in indicator or 'speech' in indicator or 'conversational' in indicator:
                    voice_ai_score += points
                signals_found.append(f"notes:{indicator}")
        
        # Normalize scores to 0-100 range
        ai_score = min(100, ai_score)
        voice_ai_score = min(100, voice_ai_score)
        
        # Determine if this is primarily a voice AI company
        voice_ai_primary = voice_ai_score >= 70
        ai_technology_user = ai_score >= 50
        automation_focus = any('automation' in signal for signal in signals_found)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'data': {
                'ai_score': ai_score,
                'voice_ai_score': voice_ai_score,
                'voice_ai_primary': voice_ai_primary,
                'ai_technology_user': ai_technology_user,
                'automation_focus': automation_focus,
                'signals_found': signals_found,
                'signal_count': len(signals_found)
            },
            'response_time_ms': response_time_ms,
            'confidence_score': min(1.0, len(signals_found) * 0.1),
            'cost_cents': 0  # Internal analysis, no external cost
        }
    
    def enrich_lead(self, company_name: str, notes: str, domain: str = None) -> Tuple[List[Dict], List[Dict], int]:
        """Enrich a lead with data from all available sources"""
        enrichment_requests = []
        enrichment_results = []
        total_cost_cents = 0
        
        # Clearbit enrichment
        clearbit_request = {
            'provider': 'clearbit',
            'company_name': company_name,
            'domain': domain,
            'enrichment_type': 'company_data',
            'timestamp': datetime.now().isoformat()
        }
        enrichment_requests.append(clearbit_request)
        
        clearbit_result = self.enrich_company_clearbit(company_name, domain)
        enrichment_results.append(clearbit_result)
        total_cost_cents += clearbit_result.get('cost_cents', 0)
        
        # AI signals detection
        clearbit_data = clearbit_result.get('data', {}) if clearbit_result.get('success') else {}
        
        ai_request = {
            'provider': 'internal_ai_detection',
            'company_name': company_name,
            'enrichment_type': 'ai_signals',
            'timestamp': datetime.now().isoformat()
        }
        enrichment_requests.append(ai_request)
        
        ai_result = self.detect_ai_signals(company_name, clearbit_data, notes)
        enrichment_results.append(ai_result)
        total_cost_cents += ai_result.get('cost_cents', 0)
        
        return enrichment_requests, enrichment_results, total_cost_cents

class EnhancedLeadQualificationModel:
    """Enhanced ML model for lead qualification with comprehensive features"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        self.model_version = "v2.0"
        self.loaded = False
        self.feature_weights = {}
        self.performance_metrics = {}
    
    def load_model(self) -> bool:
        """Load the trained model from disk"""
        try:
            model_path = os.path.join(MODELS_DIR, "qualification_model_v2.pkl")
            vectorizer_path = os.path.join(MODELS_DIR, "text_vectorizer_v2.pkl")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data['feature_names']
                    self.model_version = model_data.get('version', 'v2.0')
                    self.feature_weights = model_data.get('feature_weights', {})
                    self.performance_metrics = model_data.get('performance_metrics', {})
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.loaded = True
                logger.info(f"Loaded qualification model {self.model_version}")
                logger.info(f"Performance metrics: {self.performance_metrics}")
                return True
            else:
                logger.warning("No trained model found, using enhanced fallback scoring")
                self.loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.loaded = False
            return False
    
    def extract_enhanced_features(self, meeting_data: dict, enrichment_data: List[Dict]) -> dict:
        """Extract comprehensive features from meeting and enrichment data"""
        features = {}
        
        # Meeting-based features
        features['sentiment_score'] = meeting_data.get('sentiment_score', 5)
        features['action_items_count'] = meeting_data.get('action_items_count', 0)
        features['has_follow_up'] = int(meeting_data.get('follow_up_scheduled', False))
        features['notes_length'] = len(meeting_data.get('notes', ''))
        features['data_quality_score'] = meeting_data.get('data_quality_score', 50)
        
        # Process enrichment results
        clearbit_data = {}
        ai_signals = {}
        
        for result in enrichment_data:
            if result.get('success') and result.get('data'):
                if 'clearbit' in str(result.get('data')):
                    clearbit_data = result['data']
                elif 'ai_score' in result.get('data', {}):
                    ai_signals = result['data']
        
        # Company size and revenue features
        employees = clearbit_data.get('employees', 0)
        if isinstance(employees, str):
            employees = self._parse_employee_range(employees)
        
        features['company_size'] = employees
        features['company_size_score'] = self._score_company_size(employees)
        
        revenue = clearbit_data.get('annual_revenue', 0)
        if isinstance(revenue, str):
            revenue = self._parse_revenue_string(revenue)
        features['annual_revenue'] = revenue
        features['revenue_score'] = self._score_revenue(revenue)
        
        # Enhanced AI signals
        features['ai_score'] = ai_signals.get('ai_score', 0)
        features['voice_ai_score'] = ai_signals.get('voice_ai_score', 0)
        features['voice_ai_primary'] = int(ai_signals.get('voice_ai_primary', False))
        features['ai_technology_user'] = int(ai_signals.get('ai_technology_user', False))
        features['automation_focus'] = int(ai_signals.get('automation_focus', False))
        features['ai_signal_count'] = ai_signals.get('signal_count', 0)
        
        # Technology sophistication
        technologies = clearbit_data.get('technologies', [])
        features['tech_count'] = len(technologies)
        features['uses_modern_tech'] = int(self._has_modern_tech(technologies))
        features['uses_ai_tech'] = int(self._has_ai_tech(technologies))
        
        # Industry features
        industry = clearbit_data.get('industry', '').lower()
        features['high_value_industry'] = int(industry in [
            'software', 'technology', 'saas', 'telecommunications', 
            'healthcare', 'financial services', 'e-commerce', 'fintech'
        ])
        features['ai_friendly_industry'] = int(industry in [
            'software', 'technology', 'saas', 'telecommunications',
            'healthcare', 'customer service', 'e-commerce'
        ])
        
        # Company maturity features
        founded_year = clearbit_data.get('founded')
        if founded_year:
            current_year = datetime.now().year
            company_age = current_year - founded_year
            features['company_age'] = company_age
            features['is_startup'] = int(company_age <= 5)
            features['is_established'] = int(company_age >= 10)
        else:
            features['company_age'] = 0
            features['is_startup'] = 0
            features['is_established'] = 0
        
        # Engagement quality features
        notes = meeting_data.get('notes', '')
        features['mentions_integration'] = int('integration' in notes.lower())
        features['mentions_api'] = int('api' in notes.lower())
        features['mentions_technical'] = int(any(word in notes.lower() 
            for word in ['technical', 'developer', 'engineering', 'platform']))
        features['mentions_urgency'] = int(any(word in notes.lower() 
            for word in ['urgent', 'asap', 'immediately', 'needed yesterday']))
        features['mentions_budget'] = int(any(word in notes.lower() 
            for word in ['budget', 'approved', 'funding', 'investment']))
        
        return features
    
    def _parse_employee_range(self, employee_str: str) -> int:
        """Parse employee range string to number"""
        try:
            if '-' in employee_str:
                parts = employee_str.split('-')
                return int(parts[1])  # Take upper bound
            else:
                return int(employee_str)
        except:
            return 0
    
    def _parse_revenue_string(self, revenue_str: str) -> int:
        """Parse revenue string to integer"""
        if not revenue_str:
            return 0
        
        revenue_str = revenue_str.lower().replace(',', '').replace('$', '')
        
        try:
            if 'million' in revenue_str or 'm' in revenue_str:
                number = float(revenue_str.replace('million', '').replace('m', '').strip())
                return int(number * 1_000_000)
            elif 'billion' in revenue_str or 'b' in revenue_str:
                number = float(revenue_str.replace('billion', '').replace('b', '').strip())
                return int(number * 1_000_000_000)
            else:
                return int(float(revenue_str))
        except:
            return 0
    
    def _score_company_size(self, employees: int) -> int:
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
    
    def _score_revenue(self, revenue: int) -> int:
        """Score company based on annual revenue"""
        if revenue >= 100_000_000:
            return 5
        elif revenue >= 50_000_000:
            return 4
        elif revenue >= 10_000_000:
            return 3
        elif revenue >= 1_000_000:
            return 2
        elif revenue >= 100_000:
            return 1
        else:
            return 0
    
    def _has_modern_tech(self, technologies: List[str]) -> bool:
        """Check if company uses modern technologies"""
        modern_tech = [
            'react', 'vue', 'angular', 'node.js', 'python', 'aws', 'azure',
            'google cloud', 'kubernetes', 'docker', 'microservices', 'api'
        ]
        return any(tech.lower() in [t.lower() for t in technologies] for tech in modern_tech)
    
    def _has_ai_tech(self, technologies: List[str]) -> bool:
        """Check if company uses AI technologies"""
        ai_tech = [
            'tensorflow', 'pytorch', 'openai', 'azure cognitive', 'google ai',
            'amazon lex', 'dialogflow', 'machine learning', 'nlp', 'speech'
        ]
        return any(tech.lower() in [t.lower() for t in technologies] for tech in ai_tech)
    
    def predict_qualification_score(self, meeting_data: dict, enrichment_data: List[Dict]) -> Dict:
        """Enhanced prediction with comprehensive feature analysis"""
        try:
            # Extract enhanced features
            numerical_features = self.extract_enhanced_features(meeting_data, enrichment_data)
            text_content = self._extract_text_features(meeting_data)
            
            if self.loaded and self.model and self.vectorizer:
                # Use ML model prediction
                feature_vector = self._prepare_feature_vector(numerical_features, text_content)
                
                # Predict probability
                X = np.array(feature_vector).reshape(1, -1)
                prediction_proba = self.model.predict_proba(X)[0]
                
                # Get feature importance if available
                feature_importance = {}
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                    for i, importance in enumerate(importances):
                        if i < len(self.feature_names):
                            feature_importance[self.feature_names[i]] = float(importance)
                
                # Get prediction score (probability of high-value lead)
                high_value_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                ml_score = int(high_value_prob * 100)
                
                # Calculate voice AI specific score
                voice_ai_fit = self._calculate_voice_ai_fit(numerical_features)
                
                return {
                    'method': 'enhanced_ml_model',
                    'model_name': f'enhanced_qualification_model_{self.model_version}',
                    'model_version': self.model_version,
                    'ml_score': ml_score,
                    'voice_ai_fit_score': voice_ai_fit,
                    'fallback_score': self._calculate_enhanced_fallback_score(numerical_features, text_content),
                    'final_score': ml_score,
                    'features_used': numerical_features,
                    'feature_importance': feature_importance,
                    'confidence': float(max(prediction_proba)),
                    'prediction_probability': float(high_value_prob),
                    'model_performance_metrics': self.performance_metrics
                }
            else:
                # Use enhanced fallback rule-based scoring
                fallback_score = self._calculate_enhanced_fallback_score(numerical_features, text_content)
                voice_ai_fit = self._calculate_voice_ai_fit(numerical_features)
                
                return {
                    'method': 'enhanced_fallback_rules',
                    'model_name': 'enhanced_fallback_v2',
                    'model_version': 'enhanced_fallback_v2.0',
                    'ml_score': None,
                    'voice_ai_fit_score': voice_ai_fit,
                    'fallback_score': fallback_score,
                    'final_score': fallback_score,
                    'features_used': numerical_features,
                    'feature_importance': {},
                    'confidence': 0.75,  # Higher confidence for enhanced rules
                    'prediction_probability': fallback_score / 100.0,
                    'model_performance_metrics': {}
                }
                
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            logger.error(traceback.format_exc())
            
            # Emergency fallback
            emergency_score = meeting_data.get('sentiment_score', 5) * 10
            return {
                'method': 'emergency_fallback',
                'model_name': 'emergency_fallback',
                'model_version': 'emergency_v1',
                'ml_score': None,
                'voice_ai_fit_score': 0,
                'fallback_score': emergency_score,
                'final_score': emergency_score,
                'features_used': {},
                'feature_importance': {},
                'confidence': 0.3,
                'prediction_probability': emergency_score / 100.0,
                'model_performance_metrics': {},
                'error': str(e)
            }
    
    def _extract_text_features(self, meeting_data: dict) -> str:
        """Extract and combine text features for vectorization"""
        text_parts = []
        
        company_name = meeting_data.get('company_name', '')
        if company_name and company_name != 'Unknown Company':
            text_parts.append(company_name)
        
        notes = meeting_data.get('notes', '')
        if notes:
            text_parts.append(notes)
        
        title = meeting_data.get('title', '')
        if title:
            text_parts.append(title)
        
        return ' '.join(text_parts)
    
    def _prepare_feature_vector(self, numerical_features: Dict, text_content: str) -> List:
        """Prepare feature vector for ML model"""
        # This would need to match the exact feature engineering used in training
        # For now, return a simplified version
        feature_vector = []
        
        # Add numerical features in expected order
        for feature_name in self.feature_names:
            if feature_name.startswith('text_'):
                # Text features would be handled by vectorizer
                feature_vector.append(0)  # Placeholder
            else:
                feature_vector.append(numerical_features.get(feature_name, 0))
        
        return feature_vector
    
    def _calculate_voice_ai_fit(self, features: Dict) -> int:
        """Calculate Voice AI fit score based on features"""
        score = 0
        
        # AI signals contribute heavily
        score += features.get('voice_ai_score', 0) * 0.4
        score += features.get('ai_score', 0) * 0.2
        
        # Voice AI specific indicators
        if features.get('voice_ai_primary'):
            score += 30
        if features.get('ai_technology_user'):
            score += 15
        if features.get('automation_focus'):
            score += 10
        
        # Industry fit
        if features.get('ai_friendly_industry'):
            score += 15
        
        # Company characteristics
        if features.get('company_size_score', 0) >= 3:  # Medium+ companies
            score += 10
        if features.get('uses_ai_tech'):
            score += 15
        
        return min(100, int(score))
    
    def _calculate_enhanced_fallback_score(self, features: dict, text_content: str) -> int:
        """Calculate enhanced rule-based score"""
        score = 50  # Base score
        
        # Sentiment score influence (25% weight)
        sentiment_component = (features.get('sentiment_score', 5) - 5) * 5
        score += sentiment_component
        
        # Company characteristics (35% weight)
        size_component = features.get('company_size_score', 0) * 3
        revenue_component = features.get('revenue_score', 0) * 3
        industry_component = 8 if features.get('high_value_industry') else 0
        score += size_component + revenue_component + industry_component
        
        # AI signals (25% weight) - Enhanced weighting
        ai_component = features.get('voice_ai_score', 0) * 0.15
        general_ai_component = features.get('ai_score', 0) * 0.1
        score += ai_component + general_ai_component
        
        # Engagement quality (15% weight)
        engagement_score = 0
        if features.get('action_items_count', 0) > 2:
            engagement_score += 4
        if features.get('has_follow_up', 0):
            engagement_score += 4
        if features.get('mentions_technical'):
            engagement_score += 3
        if features.get('mentions_urgency'):
            engagement_score += 4
        score += engagement_score
        
        # Text-based bonuses
        text_lower = text_content.lower()
        
        # High-value keywords (enhanced)
        if any(keyword in text_lower for keyword in [
            'millions', '100k+', '50k+', 'enterprise', 'scale', 'urgent',
            'voice ai', 'artificial intelligence', 'automation'
        ]):
            score += 12
        
        # Technology engagement
        if any(keyword in text_lower for keyword in [
            'api', 'integration', 'platform', 'developer', 'technical',
            'implementation', 'solution'
        ]):
            score += 8
        
        # Risk factors (enhanced)
        risk_keywords = [
            ('just exploring', -8), ('no budget', -12), ('far off', -6),
            ('maybe next year', -8), ('just curious', -6), ('many vendors', -4),
            ('early stage', -4), ('thinking about', -3)
        ]
        
        for risk_keyword, penalty in risk_keywords:
            if risk_keyword in text_lower:
                score += penalty
        
        return max(0, min(100, int(score)))

class RoutingEngine:
    """Handle routing decisions based on qualification scores"""
    
    def __init__(self):
        self.routing_rules = self._load_routing_rules()
    
    def _load_routing_rules(self) -> Dict:
        """Load routing rules configuration"""
        return {
            'high_value_threshold': 80,
            'voice_ai_threshold': 85,
            'medium_value_threshold': 60,
            'disqualify_threshold': 30,
            'ae_assignment_rules': {
                'voice_ai_specialist': ['voice_ai_primary'],
                'enterprise_ae': ['high_revenue', 'large_company'],
                'general_ae': ['default']
            }
        }
    
    def make_routing_decision(self, qualification_score: int, voice_ai_fit: int, 
                            features: Dict, company_name: str) -> Dict:
        """Make routing decision based on scores and features"""
        
        routing_input = {
            'qualification_score': qualification_score,
            'voice_ai_fit_score': voice_ai_fit,
            'company_name': company_name,
            'key_features': {
                k: v for k, v in features.items() 
                if k in ['company_size_score', 'revenue_score', 'voice_ai_primary', 'ai_technology_user']
            }
        }
        
        # Determine routing decision
        if qualification_score >= self.routing_rules['high_value_threshold']:
            if voice_ai_fit >= self.routing_rules['voice_ai_threshold']:
                decision = 'AE_HANDOFF'
                priority = 'HIGH_VOICE_AI'
                assigned_to = 'voice_ai_specialist'
            elif features.get('revenue_score', 0) >= 4:
                decision = 'AE_HANDOFF'
                priority = 'HIGH_ENTERPRISE'
                assigned_to = 'enterprise_ae'
            else:
                decision = 'AE_HANDOFF'
                priority = 'HIGH'
                assigned_to = 'general_ae'
        elif qualification_score >= self.routing_rules['medium_value_threshold']:
            decision = 'SDR_FOLLOWUP'
            priority = 'MEDIUM'
            assigned_to = 'sdr_team'
        elif qualification_score >= self.routing_rules['disqualify_threshold']:
            decision = 'NURTURE'
            priority = 'LOW'
            assigned_to = 'marketing_automation'
        else:
            decision = 'DISQUALIFY'
            priority = 'NONE'
            assigned_to = 'none'
        
        # Calculate routing confidence
        confidence = self._calculate_routing_confidence(qualification_score, voice_ai_fit, features)
        
        # Business rules applied
        business_rules = self._get_applied_business_rules(decision, features)
        
        routing_decision = {
            'routing_engine': 'enhanced_rules_v2',
            'routing_decision': decision,
            'assigned_to': assigned_to,
            'priority_level': priority,
            'routing_confidence': confidence,
            'business_rules_applied': business_rules,
            'manual_override': False,
            'qualification_score': qualification_score,
            'voice_ai_fit_score': voice_ai_fit,
            'progression_probability': qualification_score / 100.0
        }
        
        return routing_decision
    
    def _calculate_routing_confidence(self, qual_score: int, voice_ai_fit: int, features: Dict) -> float:
        """Calculate confidence in routing decision"""
        confidence = 0.5  # Base confidence
        
        # Score-based confidence
        if qual_score >= 90 or qual_score <= 20:
            confidence += 0.3  # Very clear high or low scores
        elif qual_score >= 70 or qual_score <= 40:
            confidence += 0.2  # Clear scores
        
        # Feature-based confidence
        if features.get('voice_ai_primary') and voice_ai_fit >= 85:
            confidence += 0.15  # Clear Voice AI fit
        
        if features.get('revenue_score', 0) >= 4:
            confidence += 0.1  # Clear enterprise target
        
        # Data quality confidence
        data_quality = features.get('data_quality_score', 50)
        confidence += (data_quality / 100) * 0.15
        
        return min(1.0, confidence)
    
    def _get_applied_business_rules(self, decision: str, features: Dict) -> List[str]:
        """Get list of business rules that were applied"""
        rules_applied = []
        
        if decision == 'AE_HANDOFF':
            rules_applied.append('high_qualification_score')
            if features.get('voice_ai_primary'):
                rules_applied.append('voice_ai_specialization')
            if features.get('revenue_score', 0) >= 4:
                rules_applied.append('enterprise_routing')
        
        if decision == 'SDR_FOLLOWUP':
            rules_applied.append('medium_qualification_nurture')
        
        if decision == 'DISQUALIFY':
            rules_applied.append('low_qualification_disqualify')
        
        if features.get('company_size_score', 0) >= 4:
            rules_applied.append('enterprise_company_priority')
        
        return rules_applied

class EnhancedQualificationPipeline:
    """Main pipeline orchestrating the enhanced qualification process"""
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.qual_logger = QualificationLogger() if enable_logging else None
        self.enrichment_engine = EnrichmentEngine()
        self.ml_model = EnhancedLeadQualificationModel()
        self.routing_engine = RoutingEngine()
        
        # Load ML model
        self.ml_model.load_model()
        
        logger.info("Enhanced Qualification Pipeline initialized")
    
    def process_lead(self, meeting_data: Dict, log_id: str = None) -> Dict:
        """Process a single lead through the complete qualification pipeline"""
        
        company_name = meeting_data.get('company_name', 'Unknown')
        logger.info(f"Processing lead: {company_name}")
        
        results = {
            'company_name': company_name,
            'fellow_meeting_id': meeting_data.get('id'),
            'qualification_log_id': log_id,
            'pipeline_stages': {},
            'final_score': 0,
            'routing_decision': None,
            'errors': []
        }
        
        try:
            # Stage 1: Enrichment
            logger.info(f"Starting enrichment for {company_name}")
            enrichment_start = time.time()
            
            enrichment_requests, enrichment_results, total_cost = self.enrichment_engine.enrich_lead(
                company_name=company_name,
                notes=meeting_data.get('notes', ''),
                domain=None  # Could extract from company data if available
            )
            
            enrichment_duration = int((time.time() - enrichment_start) * 1000)
            
            # Log enrichment stage
            if self.enable_logging and log_id:
                self.qual_logger.log_enrichment_stage(
                    log_id=log_id,
                    enrichment_requests=enrichment_requests,
                    enrichment_results=enrichment_results,
                    total_cost_cents=total_cost
                )
            
            results['pipeline_stages']['enrichment'] = {
                'duration_ms': enrichment_duration,
                'cost_cents': total_cost,
                'successful_enrichments': len([r for r in enrichment_results if r.get('success')]),
                'total_requests': len(enrichment_requests)
            }
            
            # Stage 2: ML Scoring
            logger.info(f"Starting ML scoring for {company_name}")
            scoring_start = time.time()
            
            scoring_result = self.ml_model.predict_qualification_score(
                meeting_data=meeting_data,
                enrichment_data=enrichment_results
            )
            
            scoring_duration = int((time.time() - scoring_start) * 1000)
            
            # Prepare model input for logging
            model_input = {
                'meeting_data': meeting_data,
                'enrichment_summary': {
                    'total_enrichments': len(enrichment_results),
                    'successful_enrichments': len([r for r in enrichment_results if r.get('success')]),
                    'enrichment_providers': [r.get('provider', 'unknown') for r in enrichment_requests]
                },
                'feature_count': len(scoring_result.get('features_used', {}))
            }
            
            # Log scoring stage
            if self.enable_logging and log_id:
                self.qual_logger.log_scoring_stage(
                    log_id=log_id,
                    model_input=model_input,
                    scoring_result=scoring_result,
                    model_performance=scoring_result.get('model_performance_metrics', {})
                )
            
            results['pipeline_stages']['scoring'] = {
                'duration_ms': scoring_duration,
                'method': scoring_result.get('method'),
                'model_version': scoring_result.get('model_version'),
                'confidence': scoring_result.get('confidence')
            }
            
            results['final_score'] = scoring_result.get('final_score', 0)
            
            # Stage 3: Routing Decision
            logger.info(f"Making routing decision for {company_name}")
            routing_start = time.time()
            
            routing_decision = self.routing_engine.make_routing_decision(
                qualification_score=scoring_result.get('final_score', 0),
                voice_ai_fit=scoring_result.get('voice_ai_fit_score', 0),
                features=scoring_result.get('features_used', {}),
                company_name=company_name
            )
            
            routing_duration = int((time.time() - routing_start) * 1000)
            
            # Prepare routing input for logging
            routing_input = {
                'qualification_score': scoring_result.get('final_score', 0),
                'voice_ai_fit_score': scoring_result.get('voice_ai_fit_score', 0),
                'company_features': scoring_result.get('features_used', {}),
                'routing_thresholds': self.routing_engine.routing_rules
            }
            
            # Log routing stage
            if self.enable_logging and log_id:
                self.qual_logger.log_routing_stage(
                    log_id=log_id,
                    routing_input=routing_input,
                    routing_decision=routing_decision
                )
            
            results['pipeline_stages']['routing'] = {
                'duration_ms': routing_duration,
                'decision': routing_decision.get('routing_decision'),
                'priority': routing_decision.get('priority_level'),
                'assigned_to': routing_decision.get('assigned_to')
            }
            
            results['routing_decision'] = routing_decision
            
            logger.info(f"Completed processing for {company_name}: Score={results['final_score']}, "
                       f"Decision={routing_decision.get('routing_decision')}")
            
        except Exception as e:
            error_msg = f"Pipeline processing failed for {company_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results['errors'].append(error_msg)
        
        return results
    
    def process_batch(self, meetings: List[Dict], run_type: str = 'batch') -> Dict:
        """Process a batch of meetings through the qualification pipeline"""
        
        # Start qualification run
        if self.enable_logging:
            run_id = self.qual_logger.start_qualification_run(
                run_type=run_type,
                configuration={'batch_size': len(meetings), 'pipeline_version': '2.0'}
            )
        else:
            run_id = None
        
        # Initialize metrics
        metrics = QualificationMetrics()
        metrics.total_leads = len(meetings)
        
        results = {
            'run_id': run_id,
            'total_meetings': len(meetings),
            'processed_meetings': 0,
            'successful_qualifications': 0,
            'failed_qualifications': 0,
            'high_value_leads': [],
            'routing_summary': {},
            'total_enrichment_cost_cents': 0,
            'processing_time_ms': 0,
            'errors': []
        }
        
        batch_start = time.time()
        
        try:
            for meeting in meetings:
                try:
                    # Start lead qualification logging
                    log_id = None
                    if self.enable_logging:
                        log_id = self.qual_logger.start_lead_qualification(
                            fellow_meeting_id=meeting.get('id'),
                            company_name=meeting.get('company_name', 'Unknown'),
                            run_id=run_id
                        )
                    
                    # Process the lead
                    lead_result = self.process_lead(meeting, log_id=log_id)
                    
                    results['processed_meetings'] += 1
                    
                    if lead_result.get('errors'):
                        results['failed_qualifications'] += 1
                        results['errors'].extend(lead_result['errors'])
                        metrics.failed_qualifications += 1
                    else:
                        results['successful_qualifications'] += 1
                        metrics.successful_qualifications += 1
                    
                    # Track high-value leads
                    final_score = lead_result.get('final_score', 0)
                    if final_score >= 80:
                        results['high_value_leads'].append({
                            'company_name': lead_result.get('company_name'),
                            'score': final_score,
                            'routing_decision': lead_result.get('routing_decision', {}).get('routing_decision'),
                            'priority': lead_result.get('routing_decision', {}).get('priority_level')
                        })
                        metrics.high_value_leads_found += 1
                    
                    # Track routing decisions
                    routing_decision = lead_result.get('routing_decision', {}).get('routing_decision', 'unknown')
                    results['routing_summary'][routing_decision] = results['routing_summary'].get(routing_decision, 0) + 1
                    
                    # Track costs
                    enrichment_cost = lead_result.get('pipeline_stages', {}).get('enrichment', {}).get('cost_cents', 0)
                    results['total_enrichment_cost_cents'] += enrichment_cost
                    
                except Exception as e:
                    error_msg = f"Failed to process meeting {meeting.get('id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['failed_qualifications'] += 1
                    metrics.failed_qualifications += 1
        
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        finally:
            # Complete the qualification run
            batch_duration = int((time.time() - batch_start) * 1000)
            results['processing_time_ms'] = batch_duration
            
            # Calculate additional metrics
            if results['processed_meetings'] > 0:
                metrics.average_processing_time_ms = batch_duration / results['processed_meetings']
                metrics.enrichment_success_rate = results['successful_qualifications'] / results['processed_meetings']
            
            if self.enable_logging:
                error_summary = '; '.join(results['errors']) if results['errors'] else None
                self.qual_logger.complete_qualification_run(
                    run_id=run_id,
                    metrics=metrics,
                    error_summary=error_summary
                )
        
        logger.info(f"Batch processing complete: {results['successful_qualifications']}/{results['processed_meetings']} successful")
        return results

def main():
    """Main entry point for enhanced scoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Lead Qualification Pipeline')
    parser.add_argument('--mode', choices=['realtime', 'batch', 'test'], default='realtime',
                       help='Processing mode')
    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours back to process leads (realtime mode)')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable comprehensive logging')
    parser.add_argument('--test-company', type=str,
                       help='Test with specific company name')
    
    args = parser.parse_args()
    
    try:
        pipeline = EnhancedQualificationPipeline(enable_logging=not args.no_logging)
        
        if args.mode == 'test' and args.test_company:
            # Test with single company
            test_meeting = {
                'id': f'test_{int(time.time())}',
                'company_name': args.test_company,
                'title': f'Telnyx Intro Call - {args.test_company}',
                'notes': 'Discussed voice AI integration needs for their customer service platform. Technical team ready for API integration.',
                'ae_name': 'Test AE',
                'date': datetime.now().isoformat(),
                'action_items_count': 3,
                'follow_up_scheduled': True,
                'sentiment_score': 8
            }
            
            result = pipeline.process_lead(test_meeting)
            print(json.dumps(result, indent=2))
            
        elif args.mode == 'batch':
            # Get unscored meetings from database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=args.hours_back)
            
            cursor.execute('''
                SELECT m.id, m.title, m.company_name, m.date, m.ae_name, 
                       m.notes, m.action_items_count, m.follow_up_scheduled,
                       m.sentiment_score, m.created_at
                FROM meetings m
                LEFT JOIN lead_scores ls ON m.id = ls.meeting_id
                WHERE m.created_at >= ? AND ls.meeting_id IS NULL
                ORDER BY m.created_at DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.info("No unscored meetings found")
                return
            
            meetings = []
            for row in rows:
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
                    'created_at': row[9]
                }
                meetings.append(meeting)
            
            logger.info(f"Processing {len(meetings)} meetings in batch mode")
            results = pipeline.process_batch(meetings, run_type='batch_manual')
            print(json.dumps(results, indent=2))
        
        else:  # realtime mode
            # Get recently created meetings that need scoring
            # This would typically be called by the scheduler
            logger.info("Enhanced real-time scoring mode - implement based on requirements")
            print({"status": "realtime mode - implement scheduler integration"})
    
    except Exception as e:
        logger.error(f"Enhanced scoring failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()