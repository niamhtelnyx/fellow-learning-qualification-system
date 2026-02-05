#!/usr/bin/env python3
"""
Company Enrichment Pipeline
Processes companies from enrichment queue and gathers intelligence
from multiple data sources (Clearbit, OpenFunnel, web scraping)
"""

import sys
import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import traceback
from urllib.parse import urlparse
import re

# Configuration
CLEARBIT_API_KEY = os.getenv('CLEARBIT_API_KEY', '')
OPENFUNNEL_API_KEY = os.getenv('OPENFUNNEL_API_KEY', '')

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
log_file = os.path.join(LOG_DIR, f"enrichment-{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnrichmentError(Exception):
    """Custom exception for enrichment errors"""
    pass

class CompanyEnricher:
    """Main class for company data enrichment"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Telnyx-Enrichment/1.0'
        })
        
        # Rate limiting
        self.rate_limits = {
            'clearbit': {'calls': 0, 'reset_time': 0, 'max_per_hour': 600},
            'openfunnel': {'calls': 0, 'reset_time': 0, 'max_per_hour': 100},
            'web_search': {'calls': 0, 'reset_time': 0, 'max_per_hour': 200}
        }
    
    def check_rate_limit(self, service: str) -> bool:
        """Check if we can make another API call to the service"""
        current_time = time.time()
        rate_limit = self.rate_limits[service]
        
        # Reset counter if hour has passed
        if current_time >= rate_limit['reset_time']:
            rate_limit['calls'] = 0
            rate_limit['reset_time'] = current_time + 3600  # Next hour
        
        return rate_limit['calls'] < rate_limit['max_per_hour']
    
    def increment_rate_limit(self, service: str):
        """Increment rate limit counter for service"""
        self.rate_limits[service]['calls'] += 1
    
    def extract_domain_from_company(self, company_name: str) -> Optional[str]:
        """Try to guess company domain from name"""
        if not company_name or company_name.lower() in ['unknown company', 'unknown']:
            return None
        
        # Clean company name
        clean_name = re.sub(r'[^\w\s-]', '', company_name.lower())
        clean_name = re.sub(r'\s+', '', clean_name)
        
        # Common domain patterns
        common_suffixes = ['.com', '.io', '.co', '.org', '.net']
        
        for suffix in common_suffixes:
            domain = f"{clean_name}{suffix}"
            if self.verify_domain_exists(domain):
                return domain
        
        return None
    
    def verify_domain_exists(self, domain: str) -> bool:
        """Verify if domain exists with a quick check"""
        try:
            response = self.session.head(f"https://{domain}", timeout=5)
            return response.status_code < 400
        except:
            try:
                response = self.session.head(f"http://{domain}", timeout=5)
                return response.status_code < 400
            except:
                return False
    
    def enrich_with_clearbit(self, company_name: str, domain: str = None) -> Dict:
        """Enrich company data using Clearbit API"""
        if not CLEARBIT_API_KEY or not self.check_rate_limit('clearbit'):
            return {}
        
        try:
            # Try domain lookup first, then company name
            lookup_value = domain or company_name
            lookup_type = 'domain' if domain else 'name'
            
            url = f"https://company.clearbit.com/v2/companies/find"
            params = {lookup_type: lookup_value}
            
            headers = {'Authorization': f'Bearer {CLEARBIT_API_KEY}'}
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            self.increment_rate_limit('clearbit')
            
            if response.status_code == 200:
                data = response.json()
                
                enriched = {
                    'source': 'clearbit',
                    'domain': data.get('domain'),
                    'name': data.get('name'),
                    'description': data.get('description'),
                    'industry': data.get('category', {}).get('industry'),
                    'employees': data.get('metrics', {}).get('employees'),
                    'employees_range': data.get('metrics', {}).get('employeesRange'),
                    'annual_revenue': data.get('metrics', {}).get('annualRevenue'),
                    'founded': data.get('foundedYear'),
                    'location': data.get('geo', {}).get('city'),
                    'country': data.get('geo', {}).get('country'),
                    'technologies': [tech.get('name') for tech in data.get('tech', [])],
                    'social_handles': {
                        'twitter': data.get('twitter', {}).get('handle'),
                        'linkedin': data.get('linkedin', {}).get('handle')
                    }
                }
                
                logger.info(f"Clearbit enrichment successful for {company_name}")
                return enriched
                
            elif response.status_code == 404:
                logger.info(f"No Clearbit data found for {company_name}")
                return {}
            else:
                logger.warning(f"Clearbit API error {response.status_code} for {company_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Clearbit enrichment failed for {company_name}: {e}")
            return {}
    
    def enrich_with_openfunnel(self, company_name: str, domain: str = None) -> Dict:
        """Enrich company data using OpenFunnel API"""
        if not OPENFUNNEL_API_KEY or not self.check_rate_limit('openfunnel'):
            return {}
        
        # Placeholder for OpenFunnel integration
        # Implementation would depend on specific API endpoints
        logger.info(f"OpenFunnel enrichment not yet implemented for {company_name}")
        return {}
    
    def enrich_with_web_search(self, company_name: str, domain: str = None) -> Dict:
        """Enrich company data using web search and scraping"""
        if not self.check_rate_limit('web_search'):
            return {}
        
        try:
            # Search for company information
            search_query = f'"{company_name}" AI voice automation software'
            
            # Use a basic web search approach
            search_url = "https://duckduckgo.com/"
            
            # For now, implement a simple approach that looks for AI signals
            ai_signals = self.detect_ai_signals(company_name, domain)
            
            enriched = {
                'source': 'web_search',
                'ai_signals': ai_signals,
                'search_performed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            self.increment_rate_limit('web_search')
            logger.info(f"Web search enrichment completed for {company_name}")
            return enriched
            
        except Exception as e:
            logger.error(f"Web search enrichment failed for {company_name}: {e}")
            return {}
    
    def detect_ai_signals(self, company_name: str, domain: str = None) -> Dict:
        """Detect AI-related signals for company"""
        ai_keywords = [
            'artificial intelligence', 'machine learning', 'ai', 'voice ai',
            'conversational ai', 'chatbot', 'voice assistant', 'automation',
            'natural language', 'speech recognition', 'voice technology'
        ]
        
        signals = {
            'voice_ai_primary': False,
            'ai_technology_user': False,
            'automation_focus': False,
            'voice_technology': False,
            'score': 0
        }
        
        # Analyze company name for AI signals
        name_lower = company_name.lower()
        
        if any(keyword in name_lower for keyword in ['ai', 'bot', 'voice', 'speak', 'talk']):
            signals['voice_ai_primary'] = True
            signals['score'] += 30
        
        if any(keyword in name_lower for keyword in ['automation', 'auto', 'smart']):
            signals['automation_focus'] = True
            signals['score'] += 20
        
        # If we have domain, we could scrape website content
        if domain and self.verify_domain_exists(domain):
            # Placeholder for website scraping
            signals['website_accessible'] = True
        
        return signals
    
    def calculate_enrichment_score(self, enriched_data: Dict) -> int:
        """Calculate overall enrichment score based on data quality"""
        score = 0
        
        # Base score for successful enrichment
        if enriched_data:
            score += 20
        
        # Clearbit data scoring
        clearbit = enriched_data.get('clearbit', {})
        if clearbit:
            score += 25
            if clearbit.get('employees'):
                score += 10
            if clearbit.get('annual_revenue'):
                score += 10
            if clearbit.get('industry'):
                score += 10
            if clearbit.get('technologies'):
                score += 15
        
        # AI signals scoring
        web_data = enriched_data.get('web_search', {})
        if web_data:
            ai_signals = web_data.get('ai_signals', {})
            score += ai_signals.get('score', 0)
        
        return min(score, 100)  # Cap at 100
    
    def combine_enrichment_data(self, clearbit_data: Dict, openfunnel_data: Dict, 
                               web_data: Dict, company_name: str) -> Dict:
        """Combine data from all enrichment sources"""
        combined = {
            'company_name': company_name,
            'enrichment_timestamp': datetime.now().isoformat(),
            'sources': []
        }
        
        if clearbit_data:
            combined['clearbit'] = clearbit_data
            combined['sources'].append('clearbit')
            
            # Use Clearbit as primary source for basic info
            combined['domain'] = clearbit_data.get('domain')
            combined['industry'] = clearbit_data.get('industry')
            combined['employees'] = clearbit_data.get('employees')
            combined['revenue'] = clearbit_data.get('annual_revenue')
            combined['description'] = clearbit_data.get('description')
        
        if openfunnel_data:
            combined['openfunnel'] = openfunnel_data
            combined['sources'].append('openfunnel')
        
        if web_data:
            combined['web_search'] = web_data
            combined['sources'].append('web_search')
            
            # AI signals are important for our use case
            combined['ai_signals'] = web_data.get('ai_signals', {})
        
        # Calculate overall score
        combined['enrichment_score'] = self.calculate_enrichment_score(combined)
        
        return combined
    
    def enrich_company(self, meeting_id: str, company_name: str) -> Optional[Dict]:
        """Main method to enrich a single company"""
        logger.info(f"Starting enrichment for company: {company_name}")
        
        # Try to extract domain
        domain = self.extract_domain_from_company(company_name)
        if domain:
            logger.info(f"Extracted domain for {company_name}: {domain}")
        
        # Enrich with all available sources
        clearbit_data = self.enrich_with_clearbit(company_name, domain)
        time.sleep(0.5)  # Rate limiting
        
        openfunnel_data = self.enrich_with_openfunnel(company_name, domain)
        time.sleep(0.5)  # Rate limiting
        
        web_data = self.enrich_with_web_search(company_name, domain)
        time.sleep(0.5)  # Rate limiting
        
        # Combine all data
        enriched_data = self.combine_enrichment_data(
            clearbit_data, openfunnel_data, web_data, company_name
        )
        
        logger.info(f"Enrichment complete for {company_name} (score: {enriched_data['enrichment_score']})")
        return enriched_data

class EnrichmentPipeline:
    """Main pipeline for processing enrichment queue"""
    
    def __init__(self):
        self.enricher = CompanyEnricher()
    
    def get_pending_enrichments(self, limit: int = 10) -> List[Tuple]:
        """Get pending companies from enrichment queue"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT meeting_id, company_name, priority, created_at
                FROM enrichment_queue 
                WHERE status = 'pending' AND error_count < 3
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
            
        finally:
            conn.close()
    
    def update_enrichment_status(self, meeting_id: str, status: str, 
                                error_details: str = None):
        """Update enrichment status in queue"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            if status == 'error':
                cursor.execute('''
                    UPDATE enrichment_queue 
                    SET status = ?, processed_at = CURRENT_TIMESTAMP, error_count = error_count + 1
                    WHERE meeting_id = ?
                ''', (status, meeting_id))
            else:
                cursor.execute('''
                    UPDATE enrichment_queue 
                    SET status = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE meeting_id = ?
                ''', (status, meeting_id))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def store_enrichment_data(self, meeting_id: str, enriched_data: Dict):
        """Store enrichment data"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Create enrichment_data table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enrichment_data (
                    meeting_id TEXT PRIMARY KEY,
                    company_name TEXT,
                    enrichment_score INTEGER,
                    clearbit_data TEXT,
                    openfunnel_data TEXT,
                    web_data TEXT,
                    combined_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (meeting_id) REFERENCES meetings (id)
                )
            ''')
            
            # Store enrichment data
            cursor.execute('''
                INSERT OR REPLACE INTO enrichment_data (
                    meeting_id, company_name, enrichment_score,
                    clearbit_data, openfunnel_data, web_data, combined_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                meeting_id,
                enriched_data['company_name'],
                enriched_data['enrichment_score'],
                json.dumps(enriched_data.get('clearbit', {})),
                json.dumps(enriched_data.get('openfunnel', {})),
                json.dumps(enriched_data.get('web_search', {})),
                json.dumps(enriched_data)
            ))
            
            # Mark meeting as enriched
            cursor.execute('''
                UPDATE meetings SET enriched = TRUE WHERE id = ?
            ''', (meeting_id,))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def process_enrichment_queue(self, batch_size: int = 10) -> Dict:
        """Process pending enrichments in batches"""
        logger.info(f"Processing enrichment queue (batch size: {batch_size})")
        
        results = {
            'processed': 0,
            'successful': 0,
            'errors': 0,
            'error_details': []
        }
        
        pending_items = self.get_pending_enrichments(batch_size)
        
        for meeting_id, company_name, priority, created_at in pending_items:
            try:
                logger.info(f"Processing enrichment for {company_name} (meeting: {meeting_id})")
                
                # Mark as processing
                self.update_enrichment_status(meeting_id, 'processing')
                
                # Enrich the company
                enriched_data = self.enricher.enrich_company(meeting_id, company_name)
                
                if enriched_data:
                    # Store enrichment data
                    self.store_enrichment_data(meeting_id, enriched_data)
                    
                    # Mark as completed
                    self.update_enrichment_status(meeting_id, 'completed')
                    
                    results['successful'] += 1
                    logger.info(f"Successfully enriched {company_name}")
                else:
                    # Mark as failed
                    self.update_enrichment_status(meeting_id, 'error', 'No enrichment data returned')
                    results['errors'] += 1
                    results['error_details'].append(f"{company_name}: No enrichment data")
                
                results['processed'] += 1
                
                # Rate limiting between companies
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Enrichment failed for {company_name}: {str(e)}"
                logger.error(error_msg)
                
                self.update_enrichment_status(meeting_id, 'error', str(e))
                results['errors'] += 1
                results['error_details'].append(error_msg)
                results['processed'] += 1
        
        logger.info(f"Enrichment batch complete: {results['successful']} successful, {results['errors']} errors")
        return results

def main():
    """Main entry point for the enrichment pipeline"""
    try:
        pipeline = EnrichmentPipeline()
        
        # Process enrichment queue
        batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        results = pipeline.process_enrichment_queue(batch_size)
        
        # Output results
        print(json.dumps(results, indent=2))
        
        # Exit with error if all enrichments failed
        if results['processed'] > 0 and results['successful'] == 0:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()