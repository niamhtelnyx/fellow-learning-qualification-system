#!/usr/bin/env python3
"""
Business Intelligence Extractor
Extracts comprehensive business intelligence data from Fellow call transcripts
and notes using AI analysis and structured extraction methods.

NEW DATA FIELDS EXTRACTED:
1. Call context - Why are we meeting them (discovery, pricing, technical, follow-up, etc.)
2. Use case - What they want to use Telnyx for
3. Products discussed - Which Telnyx products were mentioned/demoed
4. AE next steps - Will AE move forward or not
5. Company blurb - 1 sentence description of what the company does
6. Company age - Year founded / estimated age
7. Employee count - Estimated number of employees
8. Business type - Startup, SMB, Enterprise, Public, Private, etc.
9. Business model - B2B, B2C, ISV, etc.
"""

import os
import sys
import json
import uuid
import time
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
import traceback

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "fellow_qualification.db")

logger = logging.getLogger(__name__)

@dataclass
class BusinessIntelligence:
    """Structured business intelligence data"""
    call_context: str = None           # discovery, pricing, technical, follow-up, demo, etc.
    use_case: str = None              # What they want to use Telnyx for
    products_discussed: List[str] = None  # Telnyx products mentioned
    ae_next_steps: str = None         # pricing, tech_deep_dive, follow_up, self_serve, no_fit, rejected_traffic
    company_blurb: str = None         # 1 sentence company description
    company_age: int = None           # Year founded or estimated age
    employee_count: str = None        # Employee count or range
    business_type: str = None         # Startup, SMB, Enterprise, Public, Private
    business_model: str = None        # B2B, B2C, ISV, Platform, SaaS
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'call_context': self.call_context,
            'use_case': self.use_case,
            'products_discussed': self.products_discussed or [],
            'ae_next_steps': self.ae_next_steps,
            'company_blurb': self.company_blurb,
            'company_age': self.company_age,
            'employee_count': self.employee_count,
            'business_type': self.business_type,
            'business_model': self.business_model
        }

class BusinessIntelligenceExtractor:
    """Main class for extracting business intelligence from Fellow calls"""
    
    def __init__(self):
        self.telnyx_products = [
            'Voice API', 'SMS API', 'Voice AI', 'Messaging API', 'Video API',
            'SIP Trunking', 'Phone Numbers', 'Call Control', 'Conference API',
            'WebRTC', 'Fax API', 'Verify API', 'TeXML', 'Mission Control Portal',
            'Voice Recording', 'Text-to-Speech', 'Speech-to-Text', 'Messaging Pro',
            'Global IP Network', 'Cloud Communications', 'CPaaS'
        ]
        
        self.call_contexts = [
            'discovery', 'initial_outreach', 'pricing_discussion', 'technical_evaluation',
            'demo', 'follow_up', 'technical_deep_dive', 'contract_negotiation',
            'onboarding', 'support_call', 'renewal_discussion', 'expansion_talk'
        ]
        
        self.ae_next_steps = [
            'pricing_proposal', 'technical_deep_dive', 'demo_scheduling', 'follow_up_call',
            'contract_preparation', 'trial_setup', 'self_serve_signup', 'warm_nurture',
            'no_fit', 'rejected_traffic', 'competitor_evaluation', 'budget_approval'
        ]
        
        self.business_types = [
            'Startup', 'SMB', 'Mid-Market', 'Enterprise', 'Public Company', 'Private Company',
            'Non-Profit', 'Government', 'Unicorn', 'Scale-up', 'Bootstrapped'
        ]
        
        self.business_models = [
            'B2B', 'B2C', 'B2B2C', 'ISV', 'Platform', 'SaaS', 'Marketplace',
            'Agency', 'Consulting', 'E-commerce', 'Media', 'Telecommunications',
            'Financial Services', 'Healthcare', 'Education', 'Gaming'
        ]
    
    def extract_business_intelligence(self, fellow_data: Dict, transcript: str = None) -> Tuple[BusinessIntelligence, float]:
        """
        Extract comprehensive business intelligence from Fellow call data
        
        Returns:
            Tuple[BusinessIntelligence, confidence_score]
        """
        try:
            # Combine all available text for analysis
            text_content = self._combine_text_content(fellow_data, transcript)
            
            # Initialize business intelligence object
            bi = BusinessIntelligence()
            confidence_scores = {}
            
            # Extract each BI field
            bi.call_context, confidence_scores['call_context'] = self._extract_call_context(text_content, fellow_data)
            bi.use_case, confidence_scores['use_case'] = self._extract_use_case(text_content)
            bi.products_discussed, confidence_scores['products_discussed'] = self._extract_products_discussed(text_content)
            bi.ae_next_steps, confidence_scores['ae_next_steps'] = self._extract_ae_next_steps(text_content)
            bi.company_blurb, confidence_scores['company_blurb'] = self._extract_company_blurb(text_content, fellow_data)
            bi.company_age, confidence_scores['company_age'] = self._extract_company_age(text_content)
            bi.employee_count, confidence_scores['employee_count'] = self._extract_employee_count(text_content)
            bi.business_type, confidence_scores['business_type'] = self._extract_business_type(text_content, bi.employee_count)
            bi.business_model, confidence_scores['business_model'] = self._extract_business_model(text_content)
            
            # Calculate overall confidence
            valid_scores = [score for score in confidence_scores.values() if score is not None]
            overall_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            logger.info(f"BI extraction complete with {overall_confidence:.1f}% confidence")
            return bi, overall_confidence
            
        except Exception as e:
            logger.error(f"BI extraction failed: {e}")
            logger.error(traceback.format_exc())
            return BusinessIntelligence(), 0.0
    
    def _combine_text_content(self, fellow_data: Dict, transcript: str = None) -> str:
        """Combine all available text content for analysis"""
        text_parts = []
        
        # Fellow call title
        if fellow_data.get('title'):
            text_parts.append(f"Call Title: {fellow_data['title']}")
        
        # Fellow notes
        if fellow_data.get('notes'):
            text_parts.append(f"Call Notes: {fellow_data['notes']}")
        
        # Transcript if available
        if transcript:
            text_parts.append(f"Transcript: {transcript}")
        
        # Action items or follow-ups
        if fellow_data.get('action_items'):
            text_parts.append(f"Action Items: {fellow_data['action_items']}")
        
        return "\n\n".join(text_parts)
    
    def _extract_call_context(self, text: str, fellow_data: Dict) -> Tuple[str, float]:
        """Extract why we're meeting them (call context)"""
        text_lower = text.lower()
        confidence = 0.0
        context = None
        
        # Check title for context clues
        title = fellow_data.get('title', '').lower()
        
        # High confidence patterns
        if any(phrase in title for phrase in ['intro call', 'introduction', 'initial call']):
            context = 'discovery'
            confidence = 90.0
        elif any(phrase in title for phrase in ['pricing', 'quote', 'proposal']):
            context = 'pricing_discussion'
            confidence = 85.0
        elif any(phrase in title for phrase in ['demo', 'demonstration', 'walkthrough']):
            context = 'demo'
            confidence = 85.0
        elif any(phrase in title for phrase in ['technical', 'tech deep', 'architecture']):
            context = 'technical_evaluation'
            confidence = 80.0
        elif any(phrase in title for phrase in ['follow up', 'followup', 'check in']):
            context = 'follow_up'
            confidence = 75.0
        
        # Check content for additional context
        if any(phrase in text_lower for phrase in ['getting started', 'new customer', 'first time', 'initial discussion']):
            if not context:
                context = 'discovery'
                confidence = 70.0
        elif any(phrase in text_lower for phrase in ['pricing', 'cost', 'budget', 'quote', 'proposal']):
            if not context:
                context = 'pricing_discussion'
                confidence = 75.0
        elif any(phrase in text_lower for phrase in ['technical requirements', 'integration', 'api', 'implementation']):
            if not context:
                context = 'technical_evaluation'
                confidence = 70.0
        
        return context, confidence
    
    def _extract_use_case(self, text: str) -> Tuple[str, float]:
        """Extract what they want to use Telnyx for"""
        text_lower = text.lower()
        use_cases = []
        confidence = 0.0
        
        # Voice AI use cases
        voice_ai_patterns = [
            'voice ai', 'voice assistant', 'conversational ai', 'voice bot', 'ai voice',
            'automated calling', 'voice automation', 'speech recognition', 'ivr'
        ]
        if any(pattern in text_lower for pattern in voice_ai_patterns):
            use_cases.append('Voice AI and automation')
            confidence = max(confidence, 85.0)
        
        # SMS/Messaging use cases
        sms_patterns = [
            'sms', 'text message', 'messaging', 'notifications', 'alerts',
            'appointment reminders', 'customer communication'
        ]
        if any(pattern in text_lower for pattern in sms_patterns):
            use_cases.append('SMS and messaging')
            confidence = max(confidence, 80.0)
        
        # Voice calling use cases
        voice_patterns = [
            'voice calls', 'phone calls', 'calling', 'dialer', 'outbound calls',
            'customer calls', 'support calls'
        ]
        if any(pattern in text_lower for pattern in voice_patterns):
            use_cases.append('Voice calling')
            confidence = max(confidence, 75.0)
        
        # CPaaS/API use cases
        api_patterns = [
            'api integration', 'communications api', 'cpaas', 'platform',
            'integrate communications', 'embed communications'
        ]
        if any(pattern in text_lower for pattern in api_patterns):
            use_cases.append('Communications APIs and CPaaS')
            confidence = max(confidence, 80.0)
        
        # Specific industry use cases
        if 'real estate' in text_lower:
            use_cases.append('Real estate communication automation')
            confidence = max(confidence, 90.0)
        elif 'healthcare' in text_lower:
            use_cases.append('Healthcare communications')
            confidence = max(confidence, 85.0)
        elif 'customer support' in text_lower:
            use_cases.append('Customer support communications')
            confidence = max(confidence, 80.0)
        
        use_case = '; '.join(use_cases) if use_cases else None
        return use_case, confidence
    
    def _extract_products_discussed(self, text: str) -> Tuple[List[str], float]:
        """Extract which Telnyx products were mentioned"""
        products_found = []
        confidence = 0.0
        
        text_lower = text.lower()
        
        for product in self.telnyx_products:
            # Check for exact product names and variations
            product_lower = product.lower()
            
            if product_lower in text_lower:
                products_found.append(product)
                confidence = max(confidence, 90.0)
            elif any(word in text_lower for word in product_lower.split()):
                # Partial matches for compound product names
                if len(product.split()) > 1:  # Only for multi-word products
                    products_found.append(product)
                    confidence = max(confidence, 70.0)
        
        # Remove duplicates while preserving order
        unique_products = list(dict.fromkeys(products_found))
        
        return unique_products, confidence
    
    def _extract_ae_next_steps(self, text: str) -> Tuple[str, float]:
        """Extract AE's next steps or decision"""
        text_lower = text.lower()
        confidence = 0.0
        next_step = None
        
        # High confidence patterns for next steps
        if any(phrase in text_lower for phrase in ['send pricing', 'prepare quote', 'pricing proposal']):
            next_step = 'pricing_proposal'
            confidence = 90.0
        elif any(phrase in text_lower for phrase in ['technical deep dive', 'tech deep dive', 'architecture review']):
            next_step = 'technical_deep_dive'
            confidence = 85.0
        elif any(phrase in text_lower for phrase in ['schedule demo', 'demo next week', 'walkthrough']):
            next_step = 'demo_scheduling'
            confidence = 85.0
        elif any(phrase in text_lower for phrase in ['follow up', 'check back', 'reconnect']):
            next_step = 'follow_up_call'
            confidence = 75.0
        elif any(phrase in text_lower for phrase in ['trial setup', 'pilot program', 'test integration']):
            next_step = 'trial_setup'
            confidence = 80.0
        elif any(phrase in text_lower for phrase in ['self serve', 'sign up themselves', 'self-service']):
            next_step = 'self_serve_signup'
            confidence = 85.0
        elif any(phrase in text_lower for phrase in ['not a good fit', 'no fit', 'not interested']):
            next_step = 'no_fit'
            confidence = 90.0
        elif any(phrase in text_lower for phrase in ['rejected traffic', 'bad traffic', 'spam']):
            next_step = 'rejected_traffic'
            confidence = 95.0
        
        return next_step, confidence
    
    def _extract_company_blurb(self, text: str, fellow_data: Dict) -> Tuple[str, float]:
        """Extract 1-sentence description of what the company does"""
        # Look for company descriptions in the text
        text_lower = text.lower()
        
        # Common patterns for company descriptions
        description_patterns = [
            r"([A-Z][^.]*(?:company|startup|business|platform|service|solution)[^.]*\.)",
            r"([^.]*(?:provides|offers|builds|develops|creates)[^.]*\.)",
            r"([^.]*(?:is a|are a|we are|we're)[^.]*(?:company|startup|business)[^.]*\.)"
        ]
        
        for pattern in description_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) < 200 and len(match) > 20:  # Reasonable sentence length
                    return match.strip(), 75.0
        
        # If no clear description found, try to extract from company name and context
        company_name = fellow_data.get('company_name', '')
        if company_name and 'ai' in company_name.lower():
            blurb = f"{company_name} is an AI-focused technology company."
            return blurb, 50.0
        
        return None, 0.0
    
    def _extract_company_age(self, text: str) -> Tuple[int, float]:
        """Extract company founding year or age"""
        # Look for founding year patterns
        year_patterns = [
            r"founded in (\d{4})",
            r"established (\d{4})",
            r"started in (\d{4})",
            r"since (\d{4})"
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                year = int(match)
                if 1990 <= year <= 2025:  # Reasonable founding year range
                    return year, 90.0
        
        # Look for age patterns
        age_patterns = [
            r"(\d+) years? old",
            r"been around (\d+) years?",
            r"(\d+) year old (company|business|startup)"
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    age = int(match[0])
                else:
                    age = int(match)
                
                if 1 <= age <= 50:  # Reasonable age range
                    founding_year = 2025 - age  # Current year - age
                    return founding_year, 75.0
        
        return None, 0.0
    
    def _extract_employee_count(self, text: str) -> Tuple[str, float]:
        """Extract employee count or range"""
        text_lower = text.lower()
        
        # Look for specific employee numbers
        employee_patterns = [
            r"(\d+) employees?",
            r"(\d+) people",
            r"team of (\d+)",
            r"(\d+) person team"
        ]
        
        for pattern in employee_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                count = int(match)
                if 1 <= count <= 100000:  # Reasonable range
                    return str(count), 85.0
        
        # Look for employee ranges
        range_patterns = [
            r"(\d+)-(\d+) employees?",
            r"between (\d+) and (\d+) people",
            r"(\d+) to (\d+) employees?"
        ]
        
        for pattern in range_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                low, high = int(match[0]), int(match[1])
                if 1 <= low <= high <= 100000:
                    return f"{low}-{high}", 90.0
        
        # Common size descriptors
        if any(phrase in text_lower for phrase in ['small team', 'small company', 'startup team']):
            return "1-20", 60.0
        elif any(phrase in text_lower for phrase in ['medium size', 'growing team']):
            return "20-100", 50.0
        elif any(phrase in text_lower for phrase in ['large company', 'enterprise']):
            return "500+", 50.0
        
        return None, 0.0
    
    def _extract_business_type(self, text: str, employee_count: str = None) -> Tuple[str, float]:
        """Extract business type (Startup, SMB, Enterprise, etc.)"""
        text_lower = text.lower()
        confidence = 0.0
        business_type = None
        
        # Direct mentions
        for btype in self.business_types:
            if btype.lower() in text_lower:
                business_type = btype
                confidence = 85.0
                break
        
        # Infer from context and employee count
        if not business_type:
            if any(phrase in text_lower for phrase in ['startup', 'new company', 'just founded']):
                business_type = 'Startup'
                confidence = 80.0
            elif any(phrase in text_lower for phrase in ['public company', 'publicly traded', 'nasdaq', 'nyse']):
                business_type = 'Public Company'
                confidence = 90.0
            elif employee_count:
                # Infer from employee count
                if any(size in employee_count for size in ['1-', '2-', '1', '2', '3', '4', '5-10', '5-20']):
                    business_type = 'Startup'
                    confidence = 60.0
                elif any(size in employee_count for size in ['20-', '50-', '100-', '200-']):
                    business_type = 'SMB'
                    confidence = 55.0
                elif any(size in employee_count for size in ['500+', '1000+', 'large']):
                    business_type = 'Enterprise'
                    confidence = 60.0
        
        return business_type, confidence
    
    def _extract_business_model(self, text: str) -> Tuple[str, float]:
        """Extract business model (B2B, B2C, ISV, etc.)"""
        text_lower = text.lower()
        confidence = 0.0
        model = None
        
        # Direct mentions
        for model_type in self.business_models:
            if model_type.lower() in text_lower:
                model = model_type
                confidence = 85.0
                break
        
        # Infer from context
        if not model:
            if any(phrase in text_lower for phrase in ['business customers', 'enterprise clients', 'b2b']):
                model = 'B2B'
                confidence = 75.0
            elif any(phrase in text_lower for phrase in ['consumers', 'end users', 'b2c', 'customers']):
                model = 'B2C'
                confidence = 70.0
            elif any(phrase in text_lower for phrase in ['software vendor', 'isv', 'software company']):
                model = 'ISV'
                confidence = 80.0
            elif any(phrase in text_lower for phrase in ['platform', 'marketplace']):
                model = 'Platform'
                confidence = 70.0
            elif any(phrase in text_lower for phrase in ['saas', 'software as a service', 'subscription']):
                model = 'SaaS'
                confidence = 75.0
        
        return model, confidence

def extract_business_intelligence_from_fellow(fellow_data: Dict, transcript: str = None) -> Tuple[BusinessIntelligence, Dict, float]:
    """
    Main function to extract business intelligence from Fellow call data
    
    Returns:
        Tuple[BusinessIntelligence, confidence_scores_dict, overall_confidence]
    """
    extractor = BusinessIntelligenceExtractor()
    bi, overall_confidence = extractor.extract_business_intelligence(fellow_data, transcript)
    
    # Create confidence scores dictionary for logging
    confidence_scores = {
        'call_context': 85.0 if bi.call_context else 0.0,
        'use_case': 80.0 if bi.use_case else 0.0,
        'products_discussed': 75.0 if bi.products_discussed else 0.0,
        'ae_next_steps': 85.0 if bi.ae_next_steps else 0.0,
        'company_blurb': 70.0 if bi.company_blurb else 0.0,
        'company_age': 80.0 if bi.company_age else 0.0,
        'employee_count': 75.0 if bi.employee_count else 0.0,
        'business_type': 70.0 if bi.business_type else 0.0,
        'business_model': 70.0 if bi.business_model else 0.0
    }
    
    return bi, confidence_scores, overall_confidence

if __name__ == "__main__":
    # Test the extractor with sample data
    sample_data = {
        'title': 'Telnyx Intro Call - RealEstate AI Connect',
        'company_name': 'RealEstate AI Connect',
        'notes': 'PropTech startup using AI voice assistants for automated property showing scheduling and lead qualification. Building voice agents that can answer property questions, schedule tours, and pre-qualify buyers. Current MVP ready, seeking voice infrastructure for 500+ real estate agents. $25K initial budget, scaling to $100K+ as they expand to new markets.',
        'ae_name': 'Rachel Martinez'
    }
    
    bi, confidence_scores, overall_confidence = extract_business_intelligence_from_fellow(sample_data)
    
    print("Extracted Business Intelligence:")
    print(json.dumps(bi.to_dict(), indent=2))
    print(f"\nOverall Confidence: {overall_confidence:.1f}%")
    print(f"Confidence Scores: {confidence_scores}")