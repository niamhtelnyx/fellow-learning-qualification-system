#!/usr/bin/env python3
"""
Fellow.ai Daily Data Ingestion Script
Runs daily at 6 AM CST to poll Fellow API for new call data
Part of the Fellow.ai learning qualification automation system
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

class FellowDataPipeline:
    """Main class for handling Fellow API data ingestion and processing"""
    
    def __init__(self):
        self.api_key = FELLOW_API_KEY
        self.base_url = FELLOW_BASE_URL
        self.endpoint = FELLOW_ENDPOINT
        self.session = requests.Session()
        self.session.headers.update({
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Telnyx-Fellow-Automation/1.0'
        })
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database for storing Fellow data"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create meetings table
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
                    enriched BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create enrichment_queue table
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
            
            # Create processing_log table
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
                    error_details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise FellowIngestionError(f"Database initialization error: {e}")
    
    def calculate_data_hash(self, data: dict) -> str:
        """Calculate hash of meeting data to detect changes"""
        # Create a consistent string representation for hashing
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
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise FellowIngestionError(f"Failed to fetch Fellow meetings after {max_retries} attempts: {e}")
    
    def extract_company_name(self, title: str) -> str:
        """Extract company name from meeting title"""
        import re
        # Pattern: "Telnyx Intro Call - Company Name" or "Telnyx Intro Call with ... (Company)"
        patterns = [
            r'Telnyx Intro Call - ([^(]+)(?:\([^)]*\))?',
            r'Telnyx Intro Call with .*\(([^)]+)\)',
            r'Telnyx Intro Call - (.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Company"
    
    def calculate_sentiment_score(self, notes: str, title: str) -> int:
        """Calculate sentiment score using the existing algorithm"""
        if not notes:
            return 5
        
        score = 5
        notes_lower = notes.lower()
        title_lower = title.lower()
        
        # Ultra High Value Indicators (9-10)
        if any(indicator in notes_lower for indicator in [
            '50k+', '$50', '$60', '$70', '$80', '$90', 'structurely', 
            '100k+', 'millions', 'needed months ago'
        ]):
            score = 10
        elif any(indicator in notes_lower for indicator in [
            'servicexcelerator', 'twilio displacement', 'large volume', 'enterprise scale'
        ]):
            score = 9
        # High Value (7-8)
        elif any(indicator in notes_lower for indicator in [
            'existing customer', 'ready to move', 'expansion', 
            'immediate commitment', 'multiple products'
        ]):
            score = 8
        elif any(indicator in notes_lower for indicator in [
            'integration', 'platform', 'developers ready', 'technical team', 'api integration'
        ]):
            score = 7
        # Moderate Value (6)
        elif any(indicator in notes_lower for indicator in [
            'present to leadership', 'evaluation', 'multiple vendors', 
            'partner referral', 'considering options'
        ]):
            score = 6
        
        return score
    
    def store_meeting_data(self, meetings: List[Dict]) -> tuple:
        """Store meetings in database and return counts"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        new_meetings = 0
        updated_meetings = 0
        
        try:
            for meeting in meetings:
                # Calculate data hash
                data_hash = self.calculate_data_hash(meeting)
                
                # Extract company name
                company_name = self.extract_company_name(meeting.get('title', ''))
                
                # Calculate sentiment score
                sentiment_score = self.calculate_sentiment_score(
                    meeting.get('notes', ''), 
                    meeting.get('title', '')
                )
                
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
                                updated_at = CURRENT_TIMESTAMP
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
                            raw_data, data_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        data_hash
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
    
    def log_processing_run(self, run_type: str, start_time: datetime, 
                          meetings_processed: int, new_meetings: int, 
                          errors: int, error_details: str = None):
        """Log processing run details"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_log (
                    run_type, start_time, end_time, meetings_processed,
                    new_meetings, errors, status, error_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_type,
                start_time,
                datetime.now(),
                meetings_processed,
                new_meetings,
                errors,
                'success' if errors == 0 else 'partial_failure' if new_meetings > 0 else 'failure',
                error_details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log processing run: {e}")
    
    def run_daily_ingestion(self) -> dict:
        """Main method for daily ingestion process"""
        start_time = datetime.now()
        logger.info("Starting daily Fellow data ingestion")
        
        results = {
            'status': 'success',
            'meetings_processed': 0,
            'new_meetings': 0,
            'updated_meetings': 0,
            'errors': 0,
            'error_details': None
        }
        
        try:
            # Fetch meetings from Fellow API
            meetings = self.fetch_fellow_meetings()
            results['meetings_processed'] = len(meetings)
            
            if meetings:
                # Store meetings in database
                new_count, updated_count = self.store_meeting_data(meetings)
                results['new_meetings'] = new_count
                results['updated_meetings'] = updated_count
                
                logger.info(f"Ingestion complete: {new_count} new, {updated_count} updated meetings")
            else:
                logger.info("No meetings found in specified date range")
            
        except Exception as e:
            results['status'] = 'failure'
            results['errors'] = 1
            results['error_details'] = str(e)
            logger.error(f"Daily ingestion failed: {e}")
            logger.error(traceback.format_exc())
        
        # Log the processing run
        self.log_processing_run(
            'daily_ingestion',
            start_time,
            results['meetings_processed'],
            results['new_meetings'],
            results['errors'],
            results['error_details']
        )
        
        return results

def main():
    """Main entry point for the script"""
    try:
        pipeline = FellowDataPipeline()
        results = pipeline.run_daily_ingestion()
        
        # Output results
        print(json.dumps(results, indent=2))
        
        # Exit with error code if ingestion failed
        if results['status'] == 'failure':
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()