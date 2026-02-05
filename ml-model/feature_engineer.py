#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Fellow Learning Qualification Model
Extracts and transforms company and call data into ML-ready feature vectors
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineering:
    """Feature Engineering Configuration"""
    max_text_features: int = 100
    min_df: int = 2
    max_df: float = 0.8
    ngram_range: Tuple[int, int] = (1, 2)

class CompanyFeatureExtractor:
    """Extract features from company data"""
    
    def __init__(self):
        self.industry_encoder = LabelEncoder()
        self.size_mapping = {
            'Small': 1, 'Small-Medium': 2, 'Medium': 3, 
            'Medium-Large': 4, 'Large': 5, 'Enterprise': 6
        }
        self.voice_ai_keywords = [
            'voice ai', 'conversational ai', 'ai voice', 'voice automation',
            'ai calling', 'voice agent', 'ai assistant', 'voice bot',
            'speech ai', 'voice intelligence', 'ai phone', 'automated calling'
        ]
        self.high_value_industries = [
            'Sales Automation', 'AI/Software', 'Real Estate Tech', 
            'B2B Sales', 'Call Centers', 'Healthcare Tech'
        ]
    
    def extract_website_signals(self, description: str, domain: str = None) -> Dict:
        """Extract functional signals from website content"""
        if not description:
            description = ""
        
        desc_lower = description.lower()
        
        # Authentication signals
        auth_signals = len([kw for kw in [
            'login', 'signup', 'sign up', 'register', 'account', 'user',
            'authentication', '2fa', 'verification', 'secure'
        ] if kw in desc_lower])
        
        # Transaction signals  
        transaction_signals = len([kw for kw in [
            'payment', 'checkout', 'purchase', 'buy', 'order', 'billing',
            'subscription', 'e-commerce', 'transaction', 'cart'
        ] if kw in desc_lower])
        
        # Communication signals
        comm_signals = len([kw for kw in [
            'notification', 'alert', 'message', 'sms', 'email', 'communication',
            'contact', 'support', 'customer service', 'chat'
        ] if kw in desc_lower])
        
        # Platform/B2B signals
        platform_signals = len([kw for kw in [
            'platform', 'marketplace', 'b2b', 'enterprise', 'business',
            'provider', 'customer', 'client', 'service'
        ] if kw in desc_lower])
        
        # Voice AI signals
        voice_ai_signals = len([kw for kw in self.voice_ai_keywords if kw in desc_lower])
        
        # Scale indicators
        scale_signals = len([kw for kw in [
            'millions', 'thousands', 'enterprise', 'large scale', 'high volume',
            'api', 'integration', 'developer', 'technical'
        ] if kw in desc_lower])
        
        return {
            'auth_signals': auth_signals,
            'transaction_signals': transaction_signals, 
            'communication_signals': comm_signals,
            'platform_signals': platform_signals,
            'voice_ai_signals': voice_ai_signals,
            'scale_signals': scale_signals,
            'total_functional_signals': auth_signals + transaction_signals + comm_signals + platform_signals
        }
    
    def extract_company_features(self, company_data: Dict) -> Dict:
        """Extract features from company profile data"""
        features = {}
        
        # Basic company info
        features['company_name'] = company_data.get('company', 'Unknown')
        features['domain'] = company_data.get('domain', '')
        features['industry'] = company_data.get('industry', 'Unknown')
        
        # Employee size mapping
        employee_size = company_data.get('employees', 'Unknown')
        features['employee_size_numeric'] = self.size_mapping.get(employee_size, 0)
        features['is_enterprise'] = 1 if 'Large' in str(employee_size) or 'Enterprise' in str(employee_size) else 0
        
        # Revenue indicators
        revenue = company_data.get('revenue', '')
        features['has_revenue_data'] = 1 if revenue and revenue != 'Unknown' else 0
        features['estimated_high_revenue'] = 1 if any(indicator in str(revenue) for indicator in ['$5M+', '$10M+', '$20M+']) else 0
        
        # Industry categorization
        features['high_value_industry'] = 1 if any(hv in str(company_data.get('industry', '')) for hv in self.high_value_industries) else 0
        
        # AI signals from company data
        ai_signals = company_data.get('ai_signals', '')
        features['has_ai_signals'] = 1 if ai_signals and ai_signals != 'Unknown' else 0
        features['voice_ai_primary'] = 1 if 'Voice AI' in str(ai_signals) else 0
        
        # Extract website functional signals
        notes = company_data.get('notes', '')
        description = f"{company_data.get('industry', '')} {notes}"
        website_features = self.extract_website_signals(description, features['domain'])
        features.update(website_features)
        
        return features

class CallFeatureExtractor:
    """Extract features from Fellow call data"""
    
    def __init__(self):
        self.product_keywords = {
            'voice_ai': ['voice ai', 'ai voice', 'conversational ai', 'voice automation', 'ai calling'],
            'voice': ['voice', 'calling', 'phone', 'sip', 'trunk', 'did'],
            'messaging': ['sms', 'message', 'text', 'whatsapp', 'messaging'],
            'verify': ['verify', '2fa', 'authentication', 'verification', 'otp'],
            'wireless': ['wireless', 'mobile', 'iot', 'sim', 'cellular']
        }
        
        self.urgency_keywords = [
            'asap', 'urgent', 'immediately', 'soon', 'quickly', 'rush',
            'deadline', 'timeline', 'needed yesterday', 'months ago'
        ]
        
        self.progression_keywords = {
            'positive': [
                'pricing', 'quote', 'contract', 'next steps', 'follow up',
                'technical deep dive', 'demo', 'poc', 'pilot', 'trial',
                'decision maker', 'budget', 'timeline', 'implementation'
            ],
            'negative': [
                'not interested', 'no budget', 'not now', 'thinking about it',
                'just exploring', 'early stage', 'no timeline', 'maybe later'
            ]
        }
    
    def extract_call_context(self, notes: str, title: str) -> Dict:
        """Extract context and intent from call notes"""
        if not notes:
            notes = ""
        if not title:
            title = ""
            
        combined_text = f"{title} {notes}".lower()
        
        features = {}
        
        # Product discussion analysis
        for product, keywords in self.product_keywords.items():
            count = sum(1 for kw in keywords if kw in combined_text)
            features[f'{product}_mentions'] = count
        
        features['total_products_discussed'] = sum(features[f'{prod}_mentions'] for prod in self.product_keywords.keys())
        features['primary_product_voice_ai'] = 1 if features['voice_ai_mentions'] >= 2 else 0
        
        # Urgency analysis
        urgency_count = sum(1 for kw in self.urgency_keywords if kw in combined_text)
        features['urgency_signals'] = urgency_count
        features['high_urgency'] = 1 if urgency_count >= 2 else 0
        
        # Progression analysis
        positive_signals = sum(1 for kw in self.progression_keywords['positive'] if kw in combined_text)
        negative_signals = sum(1 for kw in self.progression_keywords['negative'] if kw in combined_text)
        
        features['positive_progression_signals'] = positive_signals
        features['negative_progression_signals'] = negative_signals
        features['progression_score'] = positive_signals - negative_signals
        features['likely_progression'] = 1 if features['progression_score'] >= 2 else 0
        
        # Text length and complexity
        features['notes_length'] = len(notes)
        features['detailed_notes'] = 1 if len(notes) > 500 else 0
        
        return features
    
    def extract_outcome_features(self, call_data: Dict) -> Dict:
        """Extract actual outcome labels from call data"""
        features = {}
        
        # Direct outcome indicators
        features['follow_up_scheduled'] = call_data.get('follow_up_scheduled', 0)
        features['action_items_count'] = call_data.get('action_items_count', 0)
        features['has_action_items'] = 1 if features['action_items_count'] > 0 else 0
        
        # Sentiment/strategic scores from existing system
        features['sentiment_score'] = call_data.get('sentiment_score', 5)
        features['strategic_score'] = call_data.get('strategic_score', 5)
        features['high_sentiment'] = 1 if features['sentiment_score'] >= 7 else 0
        
        return features

class FeatureEngineeringPipeline:
    """Main feature engineering pipeline"""
    
    def __init__(self, config: FeatureEngineering = None):
        self.config = config or FeatureEngineering()
        self.company_extractor = CompanyFeatureExtractor()
        self.call_extractor = CallFeatureExtractor()
        self.text_vectorizer = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_training_data(self, call_data: pd.DataFrame, company_data: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare training data with features and labels"""
        logger.info(f"Preparing training data for {len(call_data)} calls")
        
        features_list = []
        
        for idx, call_row in call_data.iterrows():
            try:
                # Extract call features
                call_features = self.call_extractor.extract_call_context(
                    call_row.get('notes', ''),
                    call_row.get('title', '')
                )
                call_outcomes = self.call_extractor.extract_outcome_features(call_row.to_dict())
                call_features.update(call_outcomes)
                
                # Extract company features if company data is available
                company_name = call_row.get('company_name', '')
                if company_data is not None and company_name in company_data['company'].values:
                    company_row = company_data[company_data['company'] == company_name].iloc[0]
                    company_features = self.company_extractor.extract_company_features(company_row.to_dict())
                    call_features.update(company_features)
                else:
                    # Add default company features
                    default_company = {
                        'company': company_name,
                        'domain': '',
                        'industry': 'Unknown',
                        'employees': 'Unknown',
                        'revenue': 'Unknown',
                        'ai_signals': 'Unknown',
                        'notes': ''
                    }
                    company_features = self.company_extractor.extract_company_features(default_company)
                    call_features.update(company_features)
                
                # Add call metadata
                call_features['call_id'] = call_row.get('id', f'call_{idx}')
                call_features['ae_name'] = call_row.get('ae_name', 'Unknown')
                call_features['call_date'] = call_row.get('date', '')
                
                features_list.append(call_features)
                
            except Exception as e:
                logger.warning(f"Error processing call {call_row.get('id', idx)}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Created feature matrix with {len(features_df)} samples and {len(features_df.columns)} features")
        
        return features_df
    
    def create_target_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for model training"""
        # Primary target: AE progression likelihood (binary)
        features_df['target_progression'] = (
            (features_df['follow_up_scheduled'] == 1) |
            (features_df['action_items_count'] >= 2) |
            (features_df['progression_score'] >= 2) |
            (features_df['sentiment_score'] >= 8)
        ).astype(int)
        
        # Secondary target: Voice AI fit (for specialized routing)
        features_df['target_voice_ai_fit'] = (
            (features_df['voice_ai_mentions'] >= 2) |
            (features_df['voice_ai_primary'] == 1) |
            (features_df['voice_ai_signals'] >= 3)
        ).astype(int)
        
        # Qualification score (0-100)
        base_score = 50
        
        # Company factors
        base_score += features_df['employee_size_numeric'] * 5
        base_score += features_df['high_value_industry'] * 10
        base_score += features_df['has_ai_signals'] * 8
        base_score += features_df['voice_ai_primary'] * 15
        
        # Functional factors
        base_score += features_df['total_functional_signals'] * 3
        base_score += features_df['scale_signals'] * 4
        
        # Call factors  
        base_score += features_df['sentiment_score'] * 3
        base_score += features_df['progression_score'] * 5
        base_score += features_df['urgency_signals'] * 4
        
        features_df['target_qualification_score'] = np.clip(base_score, 0, 100)
        
        logger.info(f"Created target labels - Progression rate: {features_df['target_progression'].mean():.2%}")
        logger.info(f"Voice AI fit rate: {features_df['target_voice_ai_fit'].mean():.2%}")
        logger.info(f"Average qualification score: {features_df['target_qualification_score'].mean():.1f}")
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training"""
        feature_cols = [
            # Company features
            'employee_size_numeric', 'is_enterprise', 'has_revenue_data', 
            'estimated_high_revenue', 'high_value_industry', 'has_ai_signals',
            'voice_ai_primary', 'auth_signals', 'transaction_signals',
            'communication_signals', 'platform_signals', 'voice_ai_signals',
            'scale_signals', 'total_functional_signals',
            
            # Call features
            'voice_ai_mentions', 'voice_mentions', 'messaging_mentions',
            'verify_mentions', 'wireless_mentions', 'total_products_discussed',
            'primary_product_voice_ai', 'urgency_signals', 'high_urgency',
            'positive_progression_signals', 'negative_progression_signals',
            'progression_score', 'likely_progression', 'notes_length',
            'detailed_notes', 'action_items_count', 'has_action_items',
            'sentiment_score', 'strategic_score', 'high_sentiment'
        ]
        
        return feature_cols
    
    def fit_transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit transformers and transform features"""
        feature_cols = self.get_feature_columns()
        
        # Select and validate feature columns
        available_cols = [col for col in feature_cols if col in features_df.columns]
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
        
        X = features_df[available_cols].fillna(0)
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        
        self.is_fitted = True
        self.feature_columns = available_cols
        
        logger.info(f"Fitted feature engineering pipeline with {len(available_cols)} features")
        
        return X_scaled
    
    def transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample data for testing (placeholder)"""
    # Sample call data
    call_data = pd.DataFrame([
        {
            'id': 'call_1',
            'title': 'Telnyx Intro Call - Structurely',
            'company_name': 'Structurely', 
            'notes': 'Voice AI company looking for better calling infrastructure. Discussed integrating with their conversational AI platform. High volume needs, ready to move from current provider.',
            'ae_name': 'John Doe',
            'date': '2024-02-01',
            'follow_up_scheduled': 1,
            'action_items_count': 3,
            'sentiment_score': 9,
            'strategic_score': 8
        },
        {
            'id': 'call_2', 
            'title': 'Telnyx Intro Call - Chuck East',
            'company_name': 'Chuck East',
            'notes': 'Small legal practice, just looking for basic SMS functionality. No immediate timeline or budget.',
            'ae_name': 'Jane Smith',
            'date': '2024-02-01',
            'follow_up_scheduled': 0,
            'action_items_count': 0,
            'sentiment_score': 4,
            'strategic_score': 3
        }
    ])
    
    # Sample company data (from enrichment results)
    company_data = pd.DataFrame([
        {
            'company': 'Structurely',
            'domain': 'structurely.com',
            'industry': 'Sales Automation / Real Estate Tech',
            'employees': '30',
            'revenue': '$1M-$10M',
            'ai_signals': 'Voice AI Primary Business',
            'notes': 'Recent acquisition by CapStone Holdings'
        },
        {
            'company': 'Chuck East',
            'domain': 'casepocket.com', 
            'industry': 'Legal Services',
            'employees': 'Unknown',
            'revenue': 'Unknown',
            'ai_signals': 'Perfect Voice AI Fit',
            'notes': 'Early stage - scale issues'
        }
    ])
    
    return call_data, company_data

if __name__ == "__main__":
    # Test the feature engineering pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    call_data, company_data = load_sample_data()
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Prepare features
    features_df = pipeline.prepare_training_data(call_data, company_data)
    features_df = pipeline.create_target_labels(features_df)
    
    # Transform features
    X = pipeline.fit_transform(features_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature columns: {len(pipeline.feature_columns)}")
    print(f"Target progression rate: {features_df['target_progression'].mean():.2%}")
    
    # Display feature summary
    print("\nFeature Summary:")
    for col in pipeline.feature_columns[:10]:  # Show first 10 features
        if col in features_df.columns:
            print(f"  {col}: {features_df[col].describe()['mean']:.2f}")