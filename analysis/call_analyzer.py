"""
Call Analysis Engine for Fellow Learning System
Uses NLP to extract context, products discussed, and progression signals from call transcripts
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import asyncio
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from ..config.settings import (
    PRODUCT_CATEGORIES, 
    USE_CASE_SIGNALS, 
    PROGRESSION_SIGNALS,
    NLP_CONFIG
)

logger = logging.getLogger('fellow_learning.analysis')

@dataclass
class CallContext:
    """Extracted context from call analysis"""
    meeting_purpose: str
    problem_statements: List[str]
    use_cases: List[str]
    technical_requirements: Dict[str, Any]
    timeline_urgency: Optional[str]
    business_drivers: List[str]
    decision_makers: List[str]
    budget_signals: List[str]
    confidence_score: float

@dataclass
class ProductDiscussion:
    """Products and solutions discussed in the call"""
    primary_products: List[str]
    secondary_products: List[str]
    product_mentions: Dict[str, int]
    use_case_mapping: Dict[str, List[str]]
    technical_depth: Dict[str, float]
    confidence_score: float

@dataclass
class ProgressionSignals:
    """AE progression indicators from the call"""
    progression_type: str  # 'positive', 'neutral', 'negative'
    progression_confidence: float
    next_steps: List[str]
    commitment_level: str
    objections_raised: List[str]
    buying_signals: List[str]
    decision_timeline: Optional[str]
    key_quotes: List[str]

@dataclass
class CallAnalysisResult:
    """Complete analysis result for a call"""
    call_id: str
    context: CallContext
    products: ProductDiscussion
    progression: ProgressionSignals
    overall_score: float
    analysis_timestamp: datetime
    analyzer_version: str

class NLPProcessor:
    """Core NLP processing component"""
    
    def __init__(self):
        self.nlp = None
        self.sentence_model = None
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model for NER and parsing
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model")
        except OSError:
            logger.warning("spaCy model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer(NLP_CONFIG['model_name'])
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = {
            'companies': [],
            'people': [],
            'technologies': [],
            'money': [],
            'dates': [],
            'products': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['ORG']:
                entities['companies'].append(ent.text)
            elif ent.label_ in ['PERSON']:
                entities['people'].append(ent.text)
            elif ent.label_ in ['MONEY']:
                entities['money'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(ent.text)
        
        return entities
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using TF-IDF"""
        try:
            # Fit TF-IDF on the text
            sentences = self._split_into_sentences(text)
            if not sentences:
                return []
            
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get top features
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-top_n:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Failed to extract key phrases: {e}")
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

class ContextExtractor:
    """Extract call context and meeting purpose"""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp = nlp_processor
    
    def extract_context(self, transcript: str, title: str = None) -> CallContext:
        """Extract context from call transcript"""
        
        # Analyze meeting purpose
        purpose = self._identify_meeting_purpose(transcript, title)
        
        # Extract problem statements
        problems = self._extract_problem_statements(transcript)
        
        # Identify use cases
        use_cases = self._identify_use_cases(transcript)
        
        # Extract technical requirements
        tech_requirements = self._extract_technical_requirements(transcript)
        
        # Analyze timeline urgency
        urgency = self._analyze_timeline_urgency(transcript)
        
        # Extract business drivers
        drivers = self._extract_business_drivers(transcript)
        
        # Identify decision makers
        decision_makers = self._identify_decision_makers(transcript)
        
        # Detect budget signals
        budget_signals = self._detect_budget_signals(transcript)
        
        # Calculate confidence
        confidence = self._calculate_context_confidence(
            purpose, problems, use_cases, tech_requirements
        )
        
        return CallContext(
            meeting_purpose=purpose,
            problem_statements=problems,
            use_cases=use_cases,
            technical_requirements=tech_requirements,
            timeline_urgency=urgency,
            business_drivers=drivers,
            decision_makers=decision_makers,
            budget_signals=budget_signals,
            confidence_score=confidence
        )
    
    def _identify_meeting_purpose(self, transcript: str, title: str = None) -> str:
        """Identify the main purpose of the meeting"""
        text = transcript.lower()
        
        purpose_indicators = {
            'discovery': ['discovery', 'learn about', 'understand', 'explore', 'research'],
            'demo': ['demo', 'demonstration', 'show', 'walk through', 'present'],
            'pricing': ['pricing', 'cost', 'quote', 'budget', 'investment'],
            'technical': ['technical', 'integration', 'api', 'implementation'],
            'follow_up': ['follow up', 'next steps', 'check in', 'update'],
            'intro': ['introduction', 'intro', 'meet', 'getting to know']
        }
        
        purpose_scores = {}
        for purpose, indicators in purpose_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if title and purpose in title.lower():
                score += 2
            purpose_scores[purpose] = score
        
        return max(purpose_scores, key=purpose_scores.get) if purpose_scores else 'unknown'
    
    def _extract_problem_statements(self, transcript: str) -> List[str]:
        """Extract problem statements from transcript"""
        problems = []
        
        # Pattern matching for problem indicators
        problem_patterns = [
            r"(?:we're|we are) (?:struggling|having issues|facing challenges) with (.+?)[\.\,\n]",
            r"(?:the|our) (?:problem|challenge|issue) is (.+?)[\.\,\n]",
            r"(?:we need|we want|we're looking) to (?:solve|fix|address) (.+?)[\.\,\n]",
            r"(?:currently|right now) we (.+?) and (?:it's|its) (?:not working|problematic|inefficient)",
        ]
        
        for pattern in problem_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            problems.extend(matches)
        
        # Clean and deduplicate
        problems = [p.strip() for p in problems if len(p.strip()) > 10]
        return list(set(problems))
    
    def _identify_use_cases(self, transcript: str) -> List[str]:
        """Identify specific use cases mentioned"""
        use_cases = []
        text = transcript.lower()
        
        # Common use case patterns
        use_case_patterns = {
            'authentication': ['2fa', 'two factor', 'verify users', 'login verification', 'authentication'],
            'notifications': ['notifications', 'alerts', 'reminders', 'updates', 'notify customers'],
            'marketing': ['marketing campaigns', 'promotional', 'marketing messages', 'campaigns'],
            'support': ['customer support', 'help desk', 'support tickets', 'customer service'],
            'transactions': ['transaction confirmations', 'order confirmation', 'payment notification'],
            'appointments': ['appointment reminders', 'booking confirmations', 'scheduling'],
            'automation': ['automate', 'automated', 'workflow automation', 'process automation']
        }
        
        for use_case, keywords in use_case_patterns.items():
            if any(keyword in text for keyword in keywords):
                use_cases.append(use_case)
        
        return use_cases
    
    def _extract_technical_requirements(self, transcript: str) -> Dict[str, Any]:
        """Extract technical requirements mentioned"""
        text = transcript.lower()
        
        requirements = {
            'volume_mentioned': False,
            'estimated_volume': None,
            'api_required': False,
            'real_time_needed': False,
            'compliance_requirements': [],
            'integration_needs': []
        }
        
        # Volume patterns
        volume_patterns = [
            r'(\d+(?:,\d+)*)\s*(?:messages?|calls?|users?|customers?)',
            r'(?:million|thousand|hundred)\s*(?:messages?|calls?|users?)',
            r'high\s*volume',
            r'scale\s*(?:to|up)'
        ]
        
        for pattern in volume_patterns:
            if re.search(pattern, text):
                requirements['volume_mentioned'] = True
                matches = re.findall(r'(\d+(?:,\d+)*)', pattern)
                if matches:
                    requirements['estimated_volume'] = matches[0]
                break
        
        # API requirements
        if any(word in text for word in ['api', 'integration', 'webhook', 'sdk']):
            requirements['api_required'] = True
        
        # Real-time requirements
        if any(word in text for word in ['real time', 'realtime', 'instant', 'immediate']):
            requirements['real_time_needed'] = True
        
        # Compliance
        compliance_terms = ['hipaa', 'gdpr', 'pci', 'compliance', 'regulation', 'security']
        for term in compliance_terms:
            if term in text:
                requirements['compliance_requirements'].append(term)
        
        return requirements
    
    def _analyze_timeline_urgency(self, transcript: str) -> Optional[str]:
        """Analyze timeline urgency from transcript"""
        text = transcript.lower()
        
        urgency_indicators = {
            'high': ['asap', 'urgent', 'immediately', 'right away', 'this week'],
            'medium': ['next month', 'soon', 'quickly', 'in the near future'],
            'low': ['eventually', 'down the road', 'future', 'when we have time']
        }
        
        for urgency, indicators in urgency_indicators.items():
            if any(indicator in text for indicator in indicators):
                return urgency
        
        return None
    
    def _extract_business_drivers(self, transcript: str) -> List[str]:
        """Extract business drivers and motivations"""
        drivers = []
        text = transcript.lower()
        
        driver_patterns = {
            'growth': ['grow', 'scale', 'expand', 'increase'],
            'cost_reduction': ['reduce costs', 'save money', 'cheaper', 'cost effective'],
            'efficiency': ['efficiency', 'automate', 'streamline', 'optimize'],
            'customer_experience': ['customer experience', 'user experience', 'customer satisfaction'],
            'compliance': ['compliance', 'regulation', 'requirement', 'mandate']
        }
        
        for driver, keywords in driver_patterns.items():
            if any(keyword in text for keyword in keywords):
                drivers.append(driver)
        
        return drivers
    
    def _identify_decision_makers(self, transcript: str) -> List[str]:
        """Identify decision makers mentioned in the call"""
        entities = self.nlp.extract_entities(transcript)
        
        # Look for titles and roles
        decision_maker_titles = [
            'ceo', 'cto', 'cfo', 'vp', 'director', 'manager', 'head of',
            'chief', 'founder', 'owner', 'president', 'lead', 'senior'
        ]
        
        decision_makers = []
        text_lower = transcript.lower()
        
        for person in entities['people']:
            # Check if any decision maker title is near this person's name
            person_index = text_lower.find(person.lower())
            if person_index != -1:
                surrounding_text = text_lower[max(0, person_index-50):person_index+50]
                if any(title in surrounding_text for title in decision_maker_titles):
                    decision_makers.append(person)
        
        return decision_makers
    
    def _detect_budget_signals(self, transcript: str) -> List[str]:
        """Detect budget-related signals"""
        budget_signals = []
        text = transcript.lower()
        
        budget_patterns = [
            r'budget of \$?(\d+(?:,\d+)*)',
            r'approved.*\$?(\d+(?:,\d+)*)',
            r'allocated.*\$?(\d+(?:,\d+)*)',
            r'investment of \$?(\d+(?:,\d+)*)',
            'budget approved',
            'funding secured',
            'procurement process',
            'budget allocated'
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, text)
            if matches:
                budget_signals.extend(matches if isinstance(matches[0], str) else [pattern])
        
        return budget_signals
    
    def _calculate_context_confidence(self, purpose: str, problems: List[str], 
                                    use_cases: List[str], tech_req: Dict[str, Any]) -> float:
        """Calculate confidence score for context extraction"""
        confidence_factors = []
        
        if purpose != 'unknown':
            confidence_factors.append(0.3)
        if problems:
            confidence_factors.append(0.3)
        if use_cases:
            confidence_factors.append(0.2)
        if tech_req['volume_mentioned'] or tech_req['api_required']:
            confidence_factors.append(0.2)
        
        return min(sum(confidence_factors), 1.0)

class ProductDetector:
    """Detect products and solutions discussed in calls"""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp = nlp_processor
    
    def detect_products(self, transcript: str) -> ProductDiscussion:
        """Detect products discussed in the call"""
        text = transcript.lower()
        
        # Count product mentions
        product_mentions = {}
        for product, keywords in PRODUCT_CATEGORIES.items():
            mentions = 0
            for keyword in keywords:
                mentions += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            if mentions > 0:
                product_mentions[product] = mentions
        
        # Rank products by mentions and context
        primary_products = []
        secondary_products = []
        
        sorted_products = sorted(product_mentions.items(), key=lambda x: x[1], reverse=True)
        
        for product, count in sorted_products:
            if count >= 3:  # Strong mention threshold
                primary_products.append(product)
            elif count >= 1:
                secondary_products.append(product)
        
        # Map use cases to products
        use_case_mapping = self._map_use_cases_to_products(transcript, product_mentions)
        
        # Analyze technical depth
        technical_depth = self._analyze_technical_depth(transcript, product_mentions)
        
        # Calculate confidence
        confidence = self._calculate_product_confidence(product_mentions, use_case_mapping)
        
        return ProductDiscussion(
            primary_products=primary_products,
            secondary_products=secondary_products,
            product_mentions=product_mentions,
            use_case_mapping=use_case_mapping,
            technical_depth=technical_depth,
            confidence_score=confidence
        )
    
    def _map_use_cases_to_products(self, transcript: str, product_mentions: Dict[str, int]) -> Dict[str, List[str]]:
        """Map specific use cases to products"""
        use_case_mapping = {}
        
        # Define use case to product mappings
        use_case_products = {
            'Voice AI': ['conversational ai', 'voice assistant', 'ai calling', 'voice automation'],
            'Voice': ['phone calls', 'calling', 'telephony', 'voice communication'],
            'Messaging': ['sms', 'text messages', 'messaging', 'notifications'],
            'Verify': ['2fa', 'verification', 'authentication', 'otp']
        }
        
        for product, use_cases in use_case_products.items():
            if product in product_mentions:
                mentioned_use_cases = []
                for use_case in use_cases:
                    if use_case in transcript.lower():
                        mentioned_use_cases.append(use_case)
                if mentioned_use_cases:
                    use_case_mapping[product] = mentioned_use_cases
        
        return use_case_mapping
    
    def _analyze_technical_depth(self, transcript: str, product_mentions: Dict[str, int]) -> Dict[str, float]:
        """Analyze depth of technical discussion for each product"""
        technical_depth = {}
        
        technical_indicators = [
            'api', 'integration', 'webhook', 'sdk', 'implementation',
            'architecture', 'scalability', 'performance', 'latency',
            'throughput', 'redundancy', 'failover', 'monitoring'
        ]
        
        for product in product_mentions:
            # Count technical terms mentioned near product
            depth_score = 0
            for indicator in technical_indicators:
                if indicator in transcript.lower():
                    depth_score += 1
            
            # Normalize to 0-1 scale
            technical_depth[product] = min(depth_score / len(technical_indicators), 1.0)
        
        return technical_depth
    
    def _calculate_product_confidence(self, product_mentions: Dict[str, int], 
                                    use_case_mapping: Dict[str, List[str]]) -> float:
        """Calculate confidence in product detection"""
        confidence_factors = []
        
        if product_mentions:
            confidence_factors.append(0.4)
        if len(product_mentions) > 1:
            confidence_factors.append(0.2)
        if use_case_mapping:
            confidence_factors.append(0.3)
        if max(product_mentions.values()) > 3:
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)

class ProgressionAnalyzer:
    """Analyze AE progression signals from calls"""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp = nlp_processor
    
    def analyze_progression(self, transcript: str) -> ProgressionSignals:
        """Analyze progression signals in the call"""
        
        # Classify progression type
        progression_type, confidence = self._classify_progression(transcript)
        
        # Extract next steps
        next_steps = self._extract_next_steps(transcript)
        
        # Assess commitment level
        commitment = self._assess_commitment_level(transcript)
        
        # Identify objections
        objections = self._identify_objections(transcript)
        
        # Detect buying signals
        buying_signals = self._detect_buying_signals(transcript)
        
        # Estimate decision timeline
        timeline = self._estimate_decision_timeline(transcript)
        
        # Extract key quotes
        key_quotes = self._extract_key_quotes(transcript, progression_type)
        
        return ProgressionSignals(
            progression_type=progression_type,
            progression_confidence=confidence,
            next_steps=next_steps,
            commitment_level=commitment,
            objections_raised=objections,
            buying_signals=buying_signals,
            decision_timeline=timeline,
            key_quotes=key_quotes
        )
    
    def _classify_progression(self, transcript: str) -> Tuple[str, float]:
        """Classify the progression type and confidence"""
        text = transcript.lower()
        
        # Count signals for each progression type
        progression_scores = {
            'positive': 0,
            'neutral': 0,
            'negative': 0
        }
        
        for progression_type, signals in PROGRESSION_SIGNALS.items():
            for signal in signals:
                count = len(re.findall(r'\b' + re.escape(signal) + r'\b', text))
                progression_scores[progression_type] += count
        
        # Determine dominant progression type
        max_score = max(progression_scores.values())
        if max_score == 0:
            return 'neutral', 0.0
        
        progression_type = max(progression_scores, key=progression_scores.get)
        confidence = min(max_score / (max_score + sum(progression_scores.values()) - max_score), 1.0)
        
        return progression_type, confidence
    
    def _extract_next_steps(self, transcript: str) -> List[str]:
        """Extract mentioned next steps"""
        next_steps = []
        
        next_step_patterns = [
            r"(?:next step|next|follow up).*?is.*?(.+?)[\.\,\n]",
            r"(?:we'll|we will|i'll|i will).*?(.+?)[\.\,\n]",
            r"(?:action item|todo|to do).*?(.+?)[\.\,\n]",
            r"(?:schedule|set up|book).*?(.+?)[\.\,\n]"
        ]
        
        for pattern in next_step_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            next_steps.extend([step.strip() for step in matches if len(step.strip()) > 5])
        
        return list(set(next_steps))
    
    def _assess_commitment_level(self, transcript: str) -> str:
        """Assess prospect's commitment level"""
        text = transcript.lower()
        
        high_commitment = ['definitely', 'absolutely', 'yes', 'confirmed', 'committed', 'approved']
        medium_commitment = ['probably', 'likely', 'interested', 'considering', 'evaluating']
        low_commitment = ['maybe', 'possibly', 'thinking about', 'not sure', 'uncertain']
        
        if any(word in text for word in high_commitment):
            return 'high'
        elif any(word in text for word in medium_commitment):
            return 'medium'
        elif any(word in text for word in low_commitment):
            return 'low'
        
        return 'unknown'
    
    def _identify_objections(self, transcript: str) -> List[str]:
        """Identify objections raised during the call"""
        objections = []
        
        objection_patterns = [
            r"(?:but|however|though).*?(.+?)[\.\,\n]",
            r"(?:concern|worried|afraid).*?(.+?)[\.\,\n]",
            r"(?:problem|issue).*?with.*?(.+?)[\.\,\n]",
            r"(?:can't|cannot|won't|will not).*?(.+?)[\.\,\n]"
        ]
        
        for pattern in objection_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            objections.extend([obj.strip() for obj in matches if len(obj.strip()) > 5])
        
        return objections
    
    def _detect_buying_signals(self, transcript: str) -> List[str]:
        """Detect buying signals in the conversation"""
        buying_signals = []
        text = transcript.lower()
        
        signal_patterns = [
            'when can we start',
            'how much does it cost',
            'what are the next steps',
            'who do we need to involve',
            'timeline for implementation',
            'contract terms',
            'pricing options',
            'procurement process'
        ]
        
        for signal in signal_patterns:
            if signal in text:
                buying_signals.append(signal)
        
        return buying_signals
    
    def _estimate_decision_timeline(self, transcript: str) -> Optional[str]:
        """Estimate decision timeline based on conversation"""
        text = transcript.lower()
        
        timeline_patterns = {
            'immediate': ['this week', 'immediately', 'asap', 'right away'],
            'short': ['this month', 'next week', 'soon', '30 days'],
            'medium': ['next quarter', '2-3 months', 'end of quarter'],
            'long': ['next year', 'eventually', '6 months', 'long term']
        }
        
        for timeline, indicators in timeline_patterns.items():
            if any(indicator in text for indicator in indicators):
                return timeline
        
        return None
    
    def _extract_key_quotes(self, transcript: str, progression_type: str) -> List[str]:
        """Extract key quotes that indicate progression"""
        sentences = self.nlp._split_into_sentences(transcript)
        key_quotes = []
        
        # Look for sentences with progression indicators
        for sentence in sentences:
            if progression_type == 'positive':
                if any(signal in sentence.lower() for signal in PROGRESSION_SIGNALS['positive']):
                    key_quotes.append(sentence.strip())
            elif progression_type == 'negative':
                if any(signal in sentence.lower() for signal in PROGRESSION_SIGNALS['negative']):
                    key_quotes.append(sentence.strip())
        
        # Return top 3 most relevant quotes
        return key_quotes[:3]

class CallAnalysisEngine:
    """Main engine that orchestrates call analysis"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.context_extractor = ContextExtractor(self.nlp_processor)
        self.product_detector = ProductDetector(self.nlp_processor)
        self.progression_analyzer = ProgressionAnalyzer(self.nlp_processor)
        self.version = "1.0.0"
    
    def analyze_call(self, call_id: str, transcript: str, title: str = None) -> CallAnalysisResult:
        """Perform complete analysis of a call"""
        logger.info(f"Analyzing call {call_id}")
        
        if not transcript or len(transcript.strip()) < 50:
            logger.warning(f"Transcript too short for analysis: {call_id}")
            return self._create_empty_result(call_id)
        
        try:
            # Extract context
            context = self.context_extractor.extract_context(transcript, title)
            
            # Detect products
            products = self.product_detector.detect_products(transcript)
            
            # Analyze progression
            progression = self.progression_analyzer.analyze_progression(transcript)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(context, products, progression)
            
            result = CallAnalysisResult(
                call_id=call_id,
                context=context,
                products=products,
                progression=progression,
                overall_score=overall_score,
                analysis_timestamp=datetime.utcnow(),
                analyzer_version=self.version
            )
            
            logger.info(f"Analysis complete for {call_id}: score {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for call {call_id}: {e}")
            return self._create_empty_result(call_id)
    
    def _calculate_overall_score(self, context: CallContext, products: ProductDiscussion, 
                               progression: ProgressionSignals) -> float:
        """Calculate overall call quality score"""
        
        # Weight factors
        context_weight = 0.3
        products_weight = 0.3
        progression_weight = 0.4
        
        # Component scores
        context_score = context.confidence_score
        products_score = products.confidence_score
        
        # Progression score based on type
        progression_score_map = {
            'positive': 1.0,
            'neutral': 0.5,
            'negative': 0.1,
            'unknown': 0.3
        }
        progression_score = (
            progression_score_map.get(progression.progression_type, 0.3) * 
            progression.progression_confidence
        )
        
        # Calculate weighted average
        overall_score = (
            context_score * context_weight +
            products_score * products_weight +
            progression_score * progression_weight
        )
        
        return round(overall_score, 4)
    
    def _create_empty_result(self, call_id: str) -> CallAnalysisResult:
        """Create empty result for failed analysis"""
        return CallAnalysisResult(
            call_id=call_id,
            context=CallContext("unknown", [], [], {}, None, [], [], [], 0.0),
            products=ProductDiscussion([], [], {}, {}, {}, 0.0),
            progression=ProgressionSignals("unknown", 0.0, [], "unknown", [], [], None, []),
            overall_score=0.0,
            analysis_timestamp=datetime.utcnow(),
            analyzer_version=self.version
        )

# Batch processing
class BatchCallAnalyzer:
    """Process multiple calls in batches"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.analyzer = CallAnalysisEngine()
    
    async def analyze_calls_batch(self, calls: List[Tuple[str, str, str]]) -> List[CallAnalysisResult]:
        """Analyze multiple calls concurrently"""
        
        async def analyze_single(call_id: str, transcript: str, title: str = None):
            return await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.analyze_call, call_id, transcript, title
            )
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def analyze_with_limit(call_data):
            async with semaphore:
                call_id, transcript, title = call_data
                return await analyze_single(call_id, transcript, title)
        
        tasks = [analyze_with_limit(call_data) for call_data in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, CallAnalysisResult)]
        logger.info(f"Successfully analyzed {len(valid_results)}/{len(calls)} calls")
        
        return valid_results

# Example usage
async def main():
    """Test the call analysis engine"""
    sample_transcript = """
    Hi John, thanks for taking the time to meet today. I understand you're looking at 
    communication solutions for your platform. Can you tell me more about what you're 
    trying to solve?
    
    Sure, we're building a marketplace that connects service providers with customers, 
    and we need to send SMS notifications for bookings, confirmations, and reminders. 
    We're also interested in voice calling capabilities and maybe some AI-powered 
    voice features for customer support.
    
    That's interesting. What's your current volume like?
    
    We're processing about 50,000 transactions per month right now, but we're expecting 
    to scale to 500,000 in the next 6 months. We need something that can handle that growth.
    
    Definitely. And what's driving this urgency?
    
    We just closed our Series A and our investors want to see rapid user growth. 
    We need to implement this by end of quarter to support our expansion plans.
    
    Got it. For next steps, I'd love to set up a technical call with your engineering 
    team to discuss the API integration. How does next Tuesday work?
    
    That sounds great. I'll include our CTO on that call.
    """
    
    analyzer = CallAnalysisEngine()
    result = analyzer.analyze_call("test-call-001", sample_transcript, "Marketplace Discovery Call")
    
    print(f"Call Analysis Results:")
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Meeting Purpose: {result.context.meeting_purpose}")
    print(f"Primary Products: {result.products.primary_products}")
    print(f"Progression Type: {result.progression.progression_type}")
    print(f"Next Steps: {result.progression.next_steps}")

if __name__ == "__main__":
    asyncio.run(main())