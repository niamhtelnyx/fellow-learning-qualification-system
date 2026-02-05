#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Fellow Learning Qualification Model
Extracts and transforms company and call data into ML-ready feature vectors

ML Team: Feature Engineering Framework
- 35+ engineered features from company and call data
- Voice AI signal detection and importance weighting
- Functional classification (auth, transactions, communication)
- Text analysis and NLP preprocessing
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
    """Feature engineering configuration"""
    max_text_features: int = 100
    min_df: int = 2
    max_df: float = 0.8
    ngram_range: Tuple[int, int] = (1, 2)

class CompanyFeatureExtractor:
    """Extract predictive features from company data"""
    
    def __init__(self):
        self.industry_encoder = LabelEncoder()
        self.size_mapping = {
            'Small': 1, 'Small-Medium': 2, 'Medium': 3, 
            'Medium-Large': 4, 'Large': 5, 'Enterprise': 6,
            '1-10': 1, '10-50': 2, '50-100': 3, '100-200': 4, 
            '200-500': 5, '500+': 6
        }
        
        # Voice AI keyword patterns (highest importance features)
        self.voice_ai_keywords = [
            'voice ai', 'conversational ai', 'ai voice', 'voice automation',
            'ai calling', 'voice agent', 'ai assistant', 'voice bot',
            'speech ai', 'voice intelligence', 'ai phone', 'automated calling',
            'voice platform', 'ai conversation'
        ]
        
        # High-value industry patterns  
        self.high_value_industries = [
            'Sales Automation', 'AI/Software', 'Real Estate Tech', 
            'B2B Sales', 'Call Centers', 'Healthcare Tech', 'FinTech',
            'Conversational AI', 'Contact Center', 'Customer Service'
        ]
        
        logger.info("🏢 Company Feature Extractor initialized")
        logger.info(f"🤖 Voice AI keywords: {len(self.voice_ai_keywords)}")
        logger.info(f"🎯 High-value industries: {len(self.high_value_industries)}")
    
    def extract_website_signals(self, description: str, domain: str = None) -> Dict:
        """Extract functional business signals from company description"""
        if not description:
            description = ""
        
        desc_lower = description.lower()
        
        # Authentication function signals (indicates 2FA/verification needs)
        auth_keywords = [
            'login', 'signup', 'sign up', 'register', 'account', 'user',
            'authentication', '2fa', 'verification', 'secure', 'identity'
        ]
        auth_signals = sum(1 for kw in auth_keywords if kw in desc_lower)
        
        # Transaction function signals (indicates notification needs)
        transaction_keywords = [
            'payment', 'checkout', 'purchase', 'buy', 'order', 'billing',
            'subscription', 'e-commerce', 'transaction', 'cart', 'invoice'
        ]
        transaction_signals = sum(1 for kw in transaction_keywords if kw in desc_lower)
        
        # Communication function signals (indicates messaging needs)
        comm_keywords = [
            'notification', 'alert', 'message', 'sms', 'email', 'communication',
            'contact', 'support', 'customer service', 'chat', 'call', 'voice'
        ]
        comm_signals = sum(1 for kw in comm_keywords if kw in desc_lower)
        
        # Platform/B2B signals (indicates scale and volume)
        platform_keywords = [
            'platform', 'marketplace', 'b2b', 'enterprise', 'business',
            'provider', 'customer', 'client', 'service', 'api', 'integration'
        ]
        platform_signals = sum(1 for kw in platform_keywords if kw in desc_lower)
        
        # Voice AI specific signals (highest priority)
        voice_ai_signals = sum(1 for kw in self.voice_ai_keywords if kw in desc_lower)
        
        # Scale and technology signals
        scale_keywords = [
            'millions', 'thousands', 'enterprise', 'large scale', 'high volume',
            'api', 'integration', 'developer', 'technical', 'automation'
        ]
        scale_signals = sum(1 for kw in scale_keywords if kw in desc_lower)
        
        total_signals = auth_signals + transaction_signals + comm_signals + platform_signals
        
        logger.debug(f"🔍 Functional signals - Auth: {auth_signals}, Trans: {transaction_signals}, "
                    f"Comm: {comm_signals}, Platform: {platform_signals}, Voice AI: {voice_ai_signals}")
        
        return {
            'auth_signals': auth_signals,
            'transaction_signals': transaction_signals,
            'communication_signals': comm_signals,
            'platform_signals': platform_signals,
            'voice_ai_signals': voice_ai_signals,
            'scale_signals': scale_signals,
            'total_functional_signals': total_signals
        }
    
    def extract_company_features(self, company_data: Dict) -> Dict:
        """Extract comprehensive company feature set"""
        features = {}
        
        # Basic company metadata
        features['company_name'] = company_data.get('company', 'Unknown')
        features['domain'] = company_data.get('domain', '')
        features['industry'] = company_data.get('industry', 'Unknown')
        
        # Employee size analysis (strong predictor of volume needs)
        employee_size = str(company_data.get('employees', 'Unknown'))
        features['employee_size_numeric'] = self.size_mapping.get(employee_size, 0)
        features['is_enterprise'] = 1 if any(x in employee_size for x in ['Large', 'Enterprise', '200+', '500+']) else 0
        features['is_smb'] = 1 if any(x in employee_size for x in ['Small', '1-10', '10-50']) else 0
        
        # Revenue indicators (scale proxy)
        revenue = str(company_data.get('revenue', ''))
        features['has_revenue_data'] = 1 if revenue and revenue != 'Unknown' else 0
        features['estimated_high_revenue'] = 1 if any(indicator in revenue for indicator in ['$5M+', '$10M+', '$20M+', '$50M+']) else 0
        
        # Industry value classification
        industry = str(company_data.get('industry', ''))
        features['high_value_industry'] = 1 if any(hv in industry for hv in self.high_value_industries) else 0
        features['ai_industry'] = 1 if any(ai_term in industry.lower() for ai_term in ['ai', 'artificial intelligence', 'machine learning']) else 0
        
        # AI signals analysis (most important feature group)
        ai_signals = str(company_data.get('ai_signals', ''))
        features['has_ai_signals'] = 1 if ai_signals and ai_signals != 'Unknown' else 0
        features['voice_ai_primary'] = 1 if 'Voice AI' in ai_signals else 0
        features['strong_ai_signals'] = 1 if 'Strong' in ai_signals else 0
        
        # Website functional analysis
        notes = company_data.get('notes', '')
        description = f"{industry} {notes} {ai_signals}"
        website_features = self.extract_website_signals(description, features['domain'])
        features.update(website_features)
        
        # Technology stack indicators
        tech_indicators = ['api', 'developer', 'integration', 'sdk', 'webhook', 'rest']
        features['tech_stack_signals'] = sum(1 for tech in tech_indicators if tech in description.lower())
        
        logger.debug(f"🏢 Company features extracted for {features['company_name']}: "
                    f"Size: {features['employee_size_numeric']}, Voice AI: {features['voice_ai_signals']}")
        
        return features

class CallFeatureExtractor:
    """Extract predictive features from Fellow call data"""
    
    def __init__(self):
        # Product discussion patterns
        self.product_keywords = {
            'voice_ai': ['voice ai', 'ai voice', 'conversational ai', 'voice automation', 'ai calling', 'ai assistant'],
            'voice': ['voice', 'calling', 'phone', 'sip', 'trunk', 'did', 'dial'],
            'messaging': ['sms', 'message', 'text', 'whatsapp', 'messaging', 'mms'],
            'verify': ['verify', '2fa', 'authentication', 'verification', 'otp', 'security'],
            'wireless': ['wireless', 'mobile', 'iot', 'sim', 'cellular', 'carrier']
        }
        
        # Urgency and timeline indicators
        self.urgency_keywords = [
            'asap', 'urgent', 'immediately', 'soon', 'quickly', 'rush',
            'deadline', 'timeline', 'needed yesterday', 'months ago',
            'critical', 'priority', 'time sensitive'
        ]
        
        # Progression signal patterns
        self.progression_keywords = {
            'positive': [
                'pricing', 'quote', 'contract', 'next steps', 'follow up',
                'technical deep dive', 'demo', 'poc', 'pilot', 'trial',
                'decision maker', 'budget', 'timeline', 'implementation',
                'move forward', 'interested', 'ready', 'approved'
            ],
            'negative': [
                'not interested', 'no budget', 'not now', 'thinking about it',
                'just exploring', 'early stage', 'no timeline', 'maybe later',
                'not ready', 'no immediate need', 'price too high'
            ]
        }
        
        logger.info("📞 Call Feature Extractor initialized")
        logger.info(f"🎯 Product categories: {len(self.product_keywords)}")
        logger.info(f"📈 Progression signals: {len(self.progression_keywords['positive'])} positive, {len(self.progression_keywords['negative'])} negative")
    
    def extract_call_context(self, notes: str, title: str) -> Dict:
        """Extract call context and intent signals"""
        if not notes:
            notes = ""
        if not title:
            title = ""
        
        combined_text = f"{title} {notes}".lower()
        features = {}
        
        # Product discussion analysis (what solutions were discussed)
        for product, keywords in self.product_keywords.items():
            count = sum(1 for kw in keywords if kw in combined_text)
            features[f'{product}_mentions'] = count
        
        # Product discussion breadth
        features['total_products_discussed'] = sum(features[f'{prod}_mentions'] for prod in self.product_keywords.keys())
        features['primary_product_voice_ai'] = 1 if features['voice_ai_mentions'] >= 2 else 0
        features['multi_product_discussion'] = 1 if features['total_products_discussed'] >= 3 else 0
        
        # Urgency and timeline analysis
        urgency_count = sum(1 for kw in self.urgency_keywords if kw in combined_text)
        features['urgency_signals'] = urgency_count
        features['high_urgency'] = 1 if urgency_count >= 2 else 0
        
        # Progression likelihood analysis
        positive_signals = sum(1 for kw in self.progression_keywords['positive'] if kw in combined_text)
        negative_signals = sum(1 for kw in self.progression_keywords['negative'] if kw in combined_text)
        
        features['positive_progression_signals'] = positive_signals
        features['negative_progression_signals'] = negative_signals
        features['progression_score'] = positive_signals - negative_signals
        features['likely_progression'] = 1 if features['progression_score'] >= 2 else 0
        
        # Call depth and engagement analysis
        features['notes_length'] = len(notes)
        features['detailed_notes'] = 1 if len(notes) > 500 else 0
        features['technical_discussion'] = 1 if any(tech in combined_text for tech in ['api', 'integration', 'technical', 'developer']) else 0
        
        # Business need indicators
        business_needs = ['scale', 'growth', 'expand', 'volume', 'capacity', 'reliability']
        features['business_need_signals'] = sum(1 for need in business_needs if need in combined_text)
        
        logger.debug(f"📞 Call features - Products: {features['total_products_discussed']}, "
                    f"Progression: {features['progression_score']}, Voice AI: {features['voice_ai_mentions']}")
        
        return features
    
    def extract_outcome_features(self, call_data: Dict) -> Dict:
        """Extract actual call outcome indicators"""
        features = {}
        
        # Direct outcome measurements
        features['follow_up_scheduled'] = int(call_data.get('follow_up_scheduled', 0))
        features['action_items_count'] = int(call_data.get('action_items_count', 0))
        features['has_action_items'] = 1 if features['action_items_count'] > 0 else 0
        features['multiple_action_items'] = 1 if features['action_items_count'] >= 3 else 0
        
        # Sentiment and strategic scoring (existing Fellow metrics)
        features['sentiment_score'] = float(call_data.get('sentiment_score', 5))
        features['strategic_score'] = float(call_data.get('strategic_score', 5))
        features['high_sentiment'] = 1 if features['sentiment_score'] >= 7 else 0
        features['high_strategic'] = 1 if features['strategic_score'] >= 7 else 0
        
        # Combined engagement score
        features['engagement_score'] = (features['sentiment_score'] + features['strategic_score']) / 2
        features['high_engagement'] = 1 if features['engagement_score'] >= 7 else 0
        
        logger.debug(f"📊 Outcome features - Follow-up: {features['follow_up_scheduled']}, "
                    f"Sentiment: {features['sentiment_score']}, Actions: {features['action_items_count']}")
        
        return features

class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline for Fellow qualification model"""
    
    def __init__(self, config: FeatureEngineering = None):
        self.config = config or FeatureEngineering()
        self.company_extractor = CompanyFeatureExtractor()
        self.call_extractor = CallFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("⚙️ ML Feature Engineering Pipeline initialized")
        logger.info(f"🎯 Target: 35+ features from company and call data")
    
    def prepare_training_data(self, call_data: pd.DataFrame, company_data: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare comprehensive training dataset with engineered features"""
        logger.info(f"⚙️ Feature engineering for {len(call_data)} calls")
        
        features_list = []
        
        for idx, call_row in call_data.iterrows():
            try:
                # Extract call-based features
                call_features = self.call_extractor.extract_call_context(
                    call_row.get('notes', ''),
                    call_row.get('title', '')
                )
                
                # Extract call outcome features
                call_outcomes = self.call_extractor.extract_outcome_features(call_row.to_dict())
                call_features.update(call_outcomes)
                
                # Extract company-based features
                company_name = call_row.get('company_name', '')
                if company_data is not None and company_name in company_data['company'].values:
                    company_row = company_data[company_data['company'] == company_name].iloc[0]
                    company_features = self.company_extractor.extract_company_features(company_row.to_dict())
                    call_features.update(company_features)
                else:
                    # Create default company profile
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
                
                # Add metadata
                call_features['call_id'] = call_row.get('id', f'call_{idx}')
                call_features['ae_name'] = call_row.get('ae_name', 'Unknown')
                call_features['call_date'] = call_row.get('date', '')
                
                features_list.append(call_features)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing call {call_row.get('id', idx)}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Feature engineering complete: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df
    
    def create_target_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for model training"""
        
        # Primary target: AE progression likelihood
        features_df['target_progression'] = (
            (features_df.get('follow_up_scheduled', 0) == 1) |
            (features_df.get('action_items_count', 0) >= 2) |
            (features_df.get('progression_score', 0) >= 2) |
            (features_df.get('sentiment_score', 0) >= 8)
        ).astype(int)
        
        # Secondary target: Voice AI prospect identification
        features_df['target_voice_ai_fit'] = (
            (features_df.get('voice_ai_mentions', 0) >= 2) |
            (features_df.get('voice_ai_primary', 0) == 1) |
            (features_df.get('voice_ai_signals', 0) >= 3) |
            (features_df.get('ai_industry', 0) == 1)
        ).astype(int)
        
        # Qualification score (0-100) - weighted combination
        base_score = 50  # Neutral baseline
        
        # Company factors (40% weight)
        base_score += features_df.get('employee_size_numeric', 0) * 5
        base_score += features_df.get('high_value_industry', 0) * 12
        base_score += features_df.get('has_ai_signals', 0) * 8
        base_score += features_df.get('voice_ai_primary', 0) * 15
        
        # Functional factors (30% weight)
        base_score += features_df.get('total_functional_signals', 0) * 2
        base_score += features_df.get('scale_signals', 0) * 3
        base_score += features_df.get('tech_stack_signals', 0) * 2
        
        # Call factors (30% weight)
        base_score += features_df.get('sentiment_score', 5) * 2
        base_score += features_df.get('progression_score', 0) * 4
        base_score += features_df.get('urgency_signals', 0) * 3
        base_score += features_df.get('voice_ai_mentions', 0) * 5
        
        features_df['target_qualification_score'] = np.clip(base_score, 0, 100)
        
        progression_rate = features_df['target_progression'].mean()
        voice_ai_rate = features_df['target_voice_ai_fit'].mean()
        avg_qual_score = features_df['target_qualification_score'].mean()
        
        logger.info(f"🎯 Target labels created:")
        logger.info(f"   📈 Progression rate: {progression_rate:.2%}")
        logger.info(f"   🤖 Voice AI fit rate: {voice_ai_rate:.2%}")
        logger.info(f"   📊 Average qualification score: {avg_qual_score:.1f}/100")
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Get standardized feature column list for model training"""
        feature_cols = [
            # Company features (20)
            'employee_size_numeric', 'is_enterprise', 'is_smb', 'has_revenue_data',
            'estimated_high_revenue', 'high_value_industry', 'ai_industry',
            'has_ai_signals', 'voice_ai_primary', 'strong_ai_signals',
            'auth_signals', 'transaction_signals', 'communication_signals',
            'platform_signals', 'voice_ai_signals', 'scale_signals',
            'total_functional_signals', 'tech_stack_signals',
            
            # Call features (17)
            'voice_ai_mentions', 'voice_mentions', 'messaging_mentions',
            'verify_mentions', 'wireless_mentions', 'total_products_discussed',
            'primary_product_voice_ai', 'multi_product_discussion',
            'urgency_signals', 'high_urgency', 'positive_progression_signals',
            'negative_progression_signals', 'progression_score', 'likely_progression',
            'notes_length', 'detailed_notes', 'technical_discussion',
            'business_need_signals', 'action_items_count', 'has_action_items',
            'multiple_action_items', 'sentiment_score', 'strategic_score',
            'high_sentiment', 'high_strategic', 'engagement_score', 'high_engagement'
        ]
        
        logger.info(f"📊 Feature set: {len(feature_cols)} standardized features")
        
        return feature_cols
    
    def fit_transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit transformers and transform features for training"""
        feature_cols = self.get_feature_columns()
        
        # Validate feature columns
        available_cols = [col for col in feature_cols if col in features_df.columns]
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Missing features: {missing_cols}")
        
        # Select features and handle missing values
        X = features_df[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.is_fitted = True
        self.feature_columns = available_cols
        
        logger.info(f"✅ Feature transformation complete: {len(available_cols)} features scaled")
        
        return X_scaled
    
    def transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample data for testing and development"""
    
    # Sample Fellow call data
    call_data = pd.DataFrame([
        {
            'id': 'call_001',
            'title': 'Telnyx Intro Call - Structurely',
            'company_name': 'Structurely',
            'notes': 'Voice AI company with real estate focus. Using Twilio but experiencing reliability issues at scale. Need high-volume calling for lead qualification. Discussed API integration and custom voice flows. Technical team ready to integrate. Timeline: Q2 implementation. Budget approved.',
            'ae_name': 'John Doe',
            'date': '2024-02-01',
            'follow_up_scheduled': 1,
            'action_items_count': 3,
            'sentiment_score': 9,
            'strategic_score': 8
        },
        {
            'id': 'call_002',
            'title': 'Telnyx Intro Call - Chuck East Legal',
            'company_name': 'Chuck East',
            'notes': 'Small legal practice looking for basic SMS functionality. No immediate timeline or budget. Early stage inquiry, just exploring options.',
            'ae_name': 'Jane Smith',
            'date': '2024-02-01',
            'follow_up_scheduled': 0,
            'action_items_count': 0,
            'sentiment_score': 4,
            'strategic_score': 3
        }
    ])
    
    # Sample company enrichment data
    company_data = pd.DataFrame([
        {
            'company': 'Structurely',
            'domain': 'structurely.com',
            'industry': 'Sales Automation / Real Estate Tech',
            'employees': '30',
            'revenue': '$1M-$10M',
            'ai_signals': 'Voice AI Primary Business',
            'notes': 'Recent acquisition by CapStone Holdings. Leading conversational AI for real estate.'
        },
        {
            'company': 'Chuck East',
            'domain': 'casepocket.com',
            'industry': 'Legal Services',
            'employees': 'Small',
            'revenue': 'Unknown',
            'ai_signals': 'Unknown',
            'notes': 'Early stage - scale issues'
        }
    ])
    
    return call_data, company_data

if __name__ == "__main__":
    # Test feature engineering pipeline
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🧪 Testing Feature Engineering Pipeline")
    logger.info("=" * 50)
    
    # Load sample data
    call_data, company_data = load_sample_data()
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Process features
    features_df = pipeline.prepare_training_data(call_data, company_data)
    features_df = pipeline.create_target_labels(features_df)
    
    # Transform for ML
    X = pipeline.fit_transform(features_df)
    
    logger.info("🎯 Feature Engineering Test Results:")
    logger.info(f"   📊 Feature matrix shape: {X.shape}")
    logger.info(f"   🔢 Feature columns: {len(pipeline.feature_columns)}")
    logger.info(f"   📈 Progression rate: {features_df['target_progression'].mean():.2%}")
    logger.info(f"   🤖 Voice AI rate: {features_df['target_voice_ai_fit'].mean():.2%}")
    logger.info(f"   📊 Avg qualification score: {features_df['target_qualification_score'].mean():.1f}/100")
    
    # Show top features
    logger.info("\n🔝 Sample Feature Values:")
    for col in pipeline.feature_columns[:10]:
        if col in features_df.columns:
            avg_val = features_df[col].mean()
            logger.info(f"   {col}: {avg_val:.2f}")
    
    logger.info("✅ Feature Engineering Pipeline Test Complete")