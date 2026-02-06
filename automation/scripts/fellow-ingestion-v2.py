#!/usr/bin/env python3
"""
Enhanced Fellow.ai Data Ingestion with Production Logging
Runs daily at 6 AM CST to poll Fellow API for new call data
Integrates with comprehensive qualification logging system
"""

import sys
import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import hashlib
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from logging_system import QualificationLogger, QualificationMetrics

# Configuration
FELLOW_API_KEY = "c2e66647b10bfbc93b85cc1b05b8bc519bc61d849a09f5ac8f767fbad927dcc4"
FELLOW_BASE_URL = "https://api.fellow.app/v1"
FELLOW_ENDPOINT = "https://telnyx.fellow.app/api/v1/recordings"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(DATA_DIR, "fellow_data.db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
log_file = os.path.join(LOG_DIR, f"fellow-ingestion-{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FellowIngestionError(Exception):
    """Custom exception for Fellow ingestion errors"""
    pass

class EnhancedFellowDataPipeline:
    """Enhanced Fellow API data ingestion with comprehensive logging"""
    
    def __init__(self, enable_logging: bool = True):
        self.api_key = FELLOW_API_KEY
        self.base_url = FELLOW_BASE_URL
        self.endpoint = FELLOW_ENDPOINT
        self.session = requests.Session()
        self.session.headers.update({
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Telnyx-Fellow-Automation/2.0'
        })
        
        # Initialize logging system
        self.enable_logging = enable_logging
        self.qual_logger = QualificationLogger() if enable_logging else None
        self.current_run_id = None
        
        self.setup_legacy_database()
    
    def setup_legacy_database(self):
        """Initialize legacy SQLite database for backward compatibility"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create meetings table (legacy)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    company_name TEXT,
                    date TEXT,
                    ae_name TEXT,
                    notes TEXT,
                    action_items_count INTEGER,
                    follow_up_scheduled BOOLEAN,
                    sentiment_score REAL,
                    strategic_score REAL,
                    raw_data TEXT,
                    data_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    enriched BOOLEAN DEFAULT FALSE,
                    current_qualification_run_id TEXT,
                    last_qualification_log_id TEXT,
                    qualification_status TEXT DEFAULT 'pending',
                    total_qualification_attempts INTEGER DEFAULT 0
                )
            ''')
            
            # Create enrichment_queue table (legacy)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enrichment_queue (
                    meeting_id TEXT,
                    company_name TEXT,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (meeting_id) REFERENCES meetings (id)
                )
            ''')
            
            # Create processing_log table (legacy)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_type TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    meetings_processed INTEGER,
                    new_meetings INTEGER,
                    errors INTEGER,
                    status TEXT,
                    error_details TEXT,
                    qualification_run_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Legacy database initialized successfully")
            
        except Exception as e:
            logger.error(f"Legacy database setup failed: {e}")
            raise FellowIngestionError(f"Database initialization error: {e}")
    
    def calculate_data_hash(self, data: dict) -> str:
        """Calculate hash of meeting data to detect changes"""
        hash_data = {
            'title': data.get('title', ''),
            'notes': data.get('notes', ''),
            'date': data.get('date', ''),
            'ae_name': data.get('ae_name', '')
        }
        data_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def fetch_fellow_meetings(self, date_range: Optional[str] = None) -> List[Dict]:
        """Fetch meetings from Fellow API with retry logic"""
        if not date_range:
            # Default to last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'date_range': date_range,
            'meeting_title': 'Telnyx Intro Call'
        }
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching Fellow meetings (attempt {attempt + 1}/{max_retries})")
                response = self.session.get(self.endpoint, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                meetings = data.get('recordings', [])
                logger.info(f"Successfully fetched {len(meetings)} meetings from Fellow API")
                return meetings
                
            except requests.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise FellowIngestionError(f"Failed to fetch Fellow meetings after {max_retries} attempts: {e}")
    
    def extract_company_name(self, title: str) -> Dict[str, any]:
        """Extract company name from meeting title with confidence scoring"""
        import re
        
        extraction_methods = [
            {
                'pattern': r'Telnyx Intro Call - ([^(]+)(?:\([^)]*\))?',
                'method': 'hyphen_format',
                'confidence': 0.9
            },
            {
                'pattern': r'Telnyx Intro Call with .*\(([^)]+)\)',
                'method': 'parentheses_format',
                'confidence': 0.85
            },
            {
                'pattern': r'Telnyx Intro Call - (.+)',
                'method': 'simple_hyphen',
                'confidence': 0.7
            },
            {
                'pattern': r'Intro Call.*?-\s*(.+)',
                'method': 'fallback_hyphen',
                'confidence': 0.5
            }
        ]
        
        for method in extraction_methods:
            match = re.search(method['pattern'], title, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
                if company_name and len(company_name) > 2:  # Basic validation
                    return {
                        'company_name': company_name,
                        'extraction_method': method['method'],
                        'confidence': method['confidence'],
                        'raw_title': title
                    }
        
        # Fallback
        return {
            'company_name': "Unknown Company",
            'extraction_method': 'fallback',
            'confidence': 0.1,
            'raw_title': title
        }
    
    def calculate_sentiment_score(self, notes: str, title: str) -> Dict[str, any]:
        """Enhanced sentiment calculation with detailed scoring"""
        if not notes:
            return {
                'sentiment_score': 5,
                'indicators_found': [],
                'risk_factors': [],
                'confidence': 0.3
            }
        
        score = 5
        indicators_found = []
        risk_factors = []
        notes_lower = notes.lower()
        
        # Ultra High Value Indicators (9-10)
        ultra_high_indicators = [
            ('50k+', 10), ('$50', 10), ('$60', 10), ('$70', 10), ('$80', 10), 
            ('$90', 10), ('structurely', 10), ('100k+', 10), ('millions', 10), 
            ('needed months ago', 9), ('urgent implementation', 9)
        ]
        
        for indicator, value in ultra_high_indicators:
            if indicator in notes_lower:
                score = max(score, value)
                indicators_found.append(indicator)
        
        # High Value (7-8)
        high_value_indicators = [
            ('existing customer', 8), ('ready to move', 8), ('expansion', 8),
            ('immediate commitment', 8), ('multiple products', 8),
            ('integration', 7), ('platform', 7), ('developers ready', 7),
            ('technical team', 7), ('api integration', 7)
        ]
        
        for indicator, value in high_value_indicators:
            if indicator in notes_lower:
                score = max(score, value)
                indicators_found.append(indicator)
        
        # Moderate Value (6)
        moderate_indicators = [
            ('present to leadership', 6), ('evaluation', 6), ('multiple vendors', 6),
            ('partner referral', 6), ('considering options', 6)
        ]
        
        for indicator, value in moderate_indicators:
            if indicator in notes_lower:
                score = max(score, value)
                indicators_found.append(indicator)
        
        # Risk factors (reduce score)
        risk_indicators = [
            ('just exploring', -2), ('no budget', -3), ('far off', -2),
            ('maybe next year', -2), ('just curious', -2), ('comparing many options', -1)
        ]
        
        for risk_factor, penalty in risk_indicators:
            if risk_factor in notes_lower:
                score = max(1, score + penalty)  # Don't go below 1
                risk_factors.append(risk_factor)
        
        # Calculate confidence based on note length and indicator count
        note_length_factor = min(1.0, len(notes) / 200)  # More text = higher confidence
        indicator_factor = min(1.0, len(indicators_found) / 3)  # More indicators = higher confidence
        confidence = (note_length_factor + indicator_factor) / 2
        
        return {
            'sentiment_score': score,
            'indicators_found': indicators_found,
            'risk_factors': risk_factors,
            'confidence': confidence,
            'note_length': len(notes)
        }
    
    def calculate_data_quality_score(self, meeting: Dict) -> float:
        """Calculate data quality score for the meeting"""
        quality_score = 0.0
        max_score = 100.0
        
        # Required fields (60% of score)
        required_fields = {
            'title': 15,
            'notes': 30,
            'ae_name': 10,
            'date': 5
        }
        
        for field, points in required_fields.items():
            value = meeting.get(field, '')
            if value and len(str(value).strip()) > 0:
                quality_score += points
        
        # Optional but valuable fields (25% of score)
        optional_fields = {
            'action_items_count': 10,
            'follow_up_scheduled': 5,
            'participant_count': 5,
            'duration': 5
        }
        
        for field, points in optional_fields.items():
            value = meeting.get(field)
            if value is not None:
                quality_score += points
        
        # Content quality (15% of score)
        notes = meeting.get('notes', '')
        if notes:
            if len(notes) >= 100:
                quality_score += 5  # Substantial notes
            if len(notes) >= 300:
                quality_score += 5  # Detailed notes
            if any(keyword in notes.lower() for keyword in ['product', 'integration', 'api', 'solution']):
                quality_score += 5  # Product-related content
        
        return min(quality_score, max_score)
    
    def process_meeting_for_logging(self, meeting: Dict) -> Dict:
        """Process a single meeting for comprehensive logging"""
        # Extract company information
        company_extraction = self.extract_company_name(meeting.get('title', ''))
        
        # Calculate sentiment with detailed analysis
        sentiment_analysis = self.calculate_sentiment_score(
            meeting.get('notes', ''), 
            meeting.get('title', '')
        )
        
        # Calculate data quality score
        data_quality = self.calculate_data_quality_score(meeting)
        
        # Prepare extracted data
        extracted_data = {
            'company_name': company_extraction['company_name'],
            'extraction_method': company_extraction['extraction_method'],
            'extraction_confidence': company_extraction['confidence'],
            'ae_name': meeting.get('ae_name'),
            'date': meeting.get('date'),
            'notes': meeting.get('notes'),
            'action_items_count': meeting.get('action_items_count', 0),
            'follow_up_scheduled': meeting.get('follow_up_scheduled', False),
            'sentiment_score': sentiment_analysis['sentiment_score'],
            'sentiment_indicators': sentiment_analysis['indicators_found'],
            'risk_factors': sentiment_analysis['risk_factors'],
            'data_quality_score': data_quality
        }
        
        return {
            'raw_fellow_data': meeting,
            'extracted_data': extracted_data,
            'data_quality_score': data_quality
        }
    
    def store_meeting_data_legacy(self, meetings: List[Dict]) -> tuple:
        """Store meetings in legacy database for backward compatibility"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        new_meetings = 0
        updated_meetings = 0
        
        try:
            for meeting in meetings:
                # Calculate data hash
                data_hash = self.calculate_data_hash(meeting)
                
                # Extract company name
                company_extraction = self.extract_company_name(meeting.get('title', ''))
                company_name = company_extraction['company_name']
                
                # Calculate sentiment score
                sentiment_analysis = self.calculate_sentiment_score(
                    meeting.get('notes', ''), 
                    meeting.get('title', '')
                )
                sentiment_score = sentiment_analysis['sentiment_score']
                
                # Check if meeting already exists
                cursor.execute('SELECT id, data_hash FROM meetings WHERE id = ?', (meeting.get('id'),))
                existing = cursor.fetchone()
                
                if existing:
                    existing_hash = existing[1]
                    if existing_hash != data_hash:
                        # Update existing meeting
                        cursor.execute('''
                            UPDATE meetings SET 
                                title = ?, company_name = ?, date = ?, ae_name = ?,
                                notes = ?, action_items_count = ?, follow_up_scheduled = ?,
                                sentiment_score = ?, raw_data = ?, data_hash = ?,
                                updated_at = CURRENT_TIMESTAMP, total_qualification_attempts = total_qualification_attempts + 1,
                                current_qualification_run_id = ?
                            WHERE id = ?
                        ''', (
                            meeting.get('title'),
                            company_name,
                            meeting.get('date'),
                            meeting.get('ae_name'),
                            meeting.get('notes'),
                            meeting.get('action_items_count', 0),
                            meeting.get('follow_up_scheduled', False),
                            sentiment_score,
                            json.dumps(meeting),
                            data_hash,
                            self.current_run_id,
                            meeting.get('id')
                        ))
                        updated_meetings += 1
                        logger.info(f"Updated meeting: {company_name}")
                else:
                    # Insert new meeting
                    cursor.execute('''
                        INSERT INTO meetings (
                            id, title, company_name, date, ae_name, notes,
                            action_items_count, follow_up_scheduled, sentiment_score,
                            raw_data, data_hash, current_qualification_run_id,
                            qualification_status, total_qualification_attempts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        meeting.get('id'),
                        meeting.get('title'),
                        company_name,
                        meeting.get('date'),
                        meeting.get('ae_name'),
                        meeting.get('notes'),
                        meeting.get('action_items_count', 0),
                        meeting.get('follow_up_scheduled', False),
                        sentiment_score,
                        json.dumps(meeting),
                        data_hash,
                        self.current_run_id,
                        'processing',
                        1
                    ))
                    
                    # Add to enrichment queue if high value
                    if sentiment_score >= 7:
                        priority = 1 if sentiment_score >= 9 else 3
                        cursor.execute('''
                            INSERT INTO enrichment_queue (meeting_id, company_name, priority)
                            VALUES (?, ?, ?)
                        ''', (meeting.get('id'), company_name, priority))
                    
                    new_meetings += 1
                    logger.info(f"Added new meeting: {company_name} (score: {sentiment_score})")
            
            conn.commit()
            return new_meetings, updated_meetings
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise FellowIngestionError(f"Failed to store meeting data: {e}")
        finally:
            conn.close()
    
    def log_processing_run_legacy(self, run_type: str, start_time: datetime, 
                                meetings_processed: int, new_meetings: int, 
                                errors: int, error_details: str = None):
        """Log processing run details to legacy table"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_log (
                    run_type, start_time, end_time, meetings_processed,
                    new_meetings, errors, status, error_details, qualification_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_type,
                start_time,
                datetime.now(),
                meetings_processed,
                new_meetings,
                errors,
                'success' if errors == 0 else 'partial_failure' if new_meetings > 0 else 'failure',
                error_details,
                self.current_run_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log processing run: {e}")
    
    def run_daily_ingestion(self, date_range: Optional[str] = None) -> dict:
        """Enhanced daily ingestion with comprehensive logging"""
        start_time = datetime.now()
        logger.info("Starting enhanced Fellow data ingestion with logging")
        
        # Initialize qualification run
        if self.enable_logging:
            self.current_run_id = self.qual_logger.start_qualification_run(
                run_type='daily_batch',
                configuration={
                    'date_range': date_range,
                    'api_endpoint': self.endpoint,
                    'pipeline_version': '2.0.0'
                }
            )
            logger.info(f"Started qualification run: {self.current_run_id}")
        
        # Initialize metrics
        metrics = QualificationMetrics()
        results = {
            'status': 'success',
            'run_id': self.current_run_id,
            'meetings_processed': 0,
            'new_meetings': 0,
            'updated_meetings': 0,
            'qualification_logs_created': 0,
            'high_value_leads': 0,
            'errors': 0,
            'error_details': []
        }
        
        try:
            # Fetch meetings from Fellow API
            meetings = self.fetch_fellow_meetings(date_range)
            results['meetings_processed'] = len(meetings)
            metrics.total_leads = len(meetings)
            
            if meetings:
                # Process each meeting
                for meeting in meetings:
                    try:
                        fellow_id = meeting.get('id')
                        
                        # Process meeting data
                        processed_meeting = self.process_meeting_for_logging(meeting)
                        extracted_data = processed_meeting['extracted_data']
                        company_name = extracted_data['company_name']
                        
                        # Start qualification logging if enabled
                        qualification_log_id = None
                        if self.enable_logging:
                            qualification_log_id = self.qual_logger.start_lead_qualification(
                                fellow_meeting_id=fellow_id,
                                company_name=company_name,
                                run_id=self.current_run_id
                            )
                            
                            # Log input capture stage
                            self.qual_logger.log_input_capture(
                                log_id=qualification_log_id,
                                raw_fellow_data=processed_meeting['raw_fellow_data'],
                                extracted_data=extracted_data,
                                data_quality_score=processed_meeting['data_quality_score']
                            )
                            
                            results['qualification_logs_created'] += 1
                        
                        # Update legacy database
                        if extracted_data['sentiment_score'] >= 8:
                            results['high_value_leads'] += 1
                            metrics.high_value_leads_found += 1
                        
                        metrics.successful_qualifications += 1
                        logger.info(f"Processed meeting: {company_name} (score: {extracted_data['sentiment_score']})")
                        
                    except Exception as e:
                        error_msg = f"Failed to process meeting {meeting.get('id', 'unknown')}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'] += 1
                        results['error_details'].append(error_msg)
                        metrics.failed_qualifications += 1
                
                # Store in legacy database for backward compatibility
                new_count, updated_count = self.store_meeting_data_legacy(meetings)
                results['new_meetings'] = new_count
                results['updated_meetings'] = updated_count
                
                logger.info(f"Ingestion complete: {new_count} new, {updated_count} updated meetings")
            else:
                logger.info("No meetings found in specified date range")
            
        except Exception as e:
            results['status'] = 'failure'
            results['errors'] += 1
            results['error_details'].append(str(e))
            metrics.failed_qualifications = metrics.total_leads
            logger.error(f"Daily ingestion failed: {e}")
            logger.error(traceback.format_exc())
        
        # Complete qualification run
        if self.enable_logging:
            error_summary = '; '.join(results['error_details']) if results['error_details'] else None
            self.qual_logger.complete_qualification_run(
                run_id=self.current_run_id,
                metrics=metrics,
                error_summary=error_summary
            )
        
        # Log to legacy system
        self.log_processing_run_legacy(
            'daily_ingestion',
            start_time,
            results['meetings_processed'],
            results['new_meetings'],
            results['errors'],
            '; '.join(results['error_details']) if results['error_details'] else None
        )
        
        return results
    
    def get_ingestion_summary(self) -> Dict:
        """Get summary of the current ingestion run"""
        if self.enable_logging and self.qual_logger and self.current_run_id:
            return self.qual_logger.get_qualification_summary(self.current_run_id)
        else:
            return {'error': 'No active qualification run or logging disabled'}

def main():
    """Main entry point for the enhanced ingestion script"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Enhanced Fellow Data Ingestion')
        parser.add_argument('--date-range', type=str, help='Date range to fetch (e.g., "2025-01-01 to 2025-01-07")')
        parser.add_argument('--no-logging', action='store_true', help='Disable comprehensive logging')
        parser.add_argument('--summary', action='store_true', help='Show ingestion summary')
        
        args = parser.parse_args()
        
        # Initialize pipeline
        pipeline = EnhancedFellowDataPipeline(enable_logging=not args.no_logging)
        
        if args.summary:
            summary = pipeline.get_ingestion_summary()
            print(json.dumps(summary, indent=2))
            return
        
        # Run ingestion
        results = pipeline.run_daily_ingestion(date_range=args.date_range)
        
        # Output results
        print(json.dumps(results, indent=2))
        
        # Show qualification summary if logging was enabled
        if not args.no_logging:
            print("\n" + "="*50)
            print("QUALIFICATION RUN SUMMARY")
            print("="*50)
            summary = pipeline.get_ingestion_summary()
            print(json.dumps(summary, indent=2))
        
        # Exit with error code if ingestion failed
        if results['status'] == 'failure':
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()