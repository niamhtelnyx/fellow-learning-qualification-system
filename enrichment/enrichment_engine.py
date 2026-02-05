"""
Company Enrichment Engine for Fellow Learning System
Orchestrates web scraping and API enrichment to build comprehensive company profiles
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import aiohttp
from urllib.parse import urljoin, urlparse
import json
import re
from bs4 import BeautifulSoup
import time

from ..config.settings import ENRICHMENT_SOURCES, PIPELINE_CONFIG, PRODUCT_CATEGORIES

logger = logging.getLogger('fellow_learning.enrichment')

@dataclass
class CompanyProfile:
    """Comprehensive company profile data"""
    name: str
    domain: str
    industry: Optional[str] = None
    sub_industry: Optional[str] = None
    employees_count: Optional[int] = None
    revenue_range: Optional[str] = None
    funding_stage: Optional[str] = None
    funding_amount: Optional[float] = None
    tech_signals: Dict[str, Any] = None
    contact_info: Dict[str, Any] = None
    social_presence: Dict[str, Any] = None
    competitive_landscape: Optional[str] = None
    enrichment_sources: List[str] = None
    enrichment_confidence: float = 0.0
    
    def __post_init__(self):
        if self.tech_signals is None:
            self.tech_signals = {}
        if self.contact_info is None:
            self.contact_info = {}
        if self.social_presence is None:
            self.social_presence = {}
        if self.enrichment_sources is None:
            self.enrichment_sources = []

class WebScraper:
    """Web scraping component for company analysis"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def scrape_company_website(self, domain: str) -> Dict[str, Any]:
        """Scrape company website for key information"""
        if not domain.startswith(('http://', 'https://')):
            domain = f"https://{domain}"
        
        try:
            # Scrape main pages
            pages_to_scrape = [
                '',  # Homepage
                '/about',
                '/about-us',
                '/company',
                '/solutions',
                '/products',
                '/api',
                '/developers',
                '/docs',
                '/pricing',
                '/contact'
            ]
            
            scraped_data = {
                'pages_found': [],
                'tech_stack': [],
                'product_mentions': {},
                'use_case_signals': {},
                'contact_info': {},
                'employee_signals': [],
                'funding_signals': [],
                'api_documentation': False,
                'developer_resources': False
            }
            
            for page_path in pages_to_scrape:
                url = urljoin(domain, page_path)
                page_data = await self._scrape_page(url)
                
                if page_data:
                    scraped_data['pages_found'].append(page_path or 'homepage')
                    self._extract_signals(page_data, scraped_data)
            
            # Calculate confidence based on pages found
            scraped_data['confidence'] = min(len(scraped_data['pages_found']) / 5, 1.0)
            
            return scraped_data
            
        except Exception as e:
            logger.error(f"Error scraping website {domain}: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single page for content"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text()
                    
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'text': ' '.join(text.split()),  # Clean whitespace
                        'links': [a.get('href') for a in soup.find_all('a', href=True)],
                        'meta_description': self._get_meta_description(soup),
                        'h1_tags': [h1.get_text().strip() for h1 in soup.find_all('h1')],
                        'h2_tags': [h2.get_text().strip() for h2 in soup.find_all('h2')]
                    }
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return None
    
    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description from page"""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            return meta.get('content', '')
        return ''
    
    def _extract_signals(self, page_data: Dict[str, Any], scraped_data: Dict[str, Any]):
        """Extract business signals from page content"""
        text = page_data['text'].lower()
        
        # Detect product categories
        for product, keywords in PRODUCT_CATEGORIES.items():
            mentions = sum(1 for keyword in keywords if keyword in text)
            if mentions > 0:
                scraped_data['product_mentions'][product] = mentions
        
        # Detect use case signals
        from ..config.settings import USE_CASE_SIGNALS
        for signal_type, keywords in USE_CASE_SIGNALS.items():
            mentions = sum(1 for keyword in keywords if keyword in text)
            if mentions > 0:
                scraped_data['use_case_signals'][signal_type] = mentions
        
        # Tech stack detection
        tech_indicators = [
            'api', 'webhook', 'sdk', 'rest', 'graphql', 'oauth',
            'aws', 'google cloud', 'azure', 'kubernetes', 'docker',
            'react', 'vue', 'angular', 'node.js', 'python', 'java',
            'mobile app', 'ios app', 'android app', 'native app'
        ]
        
        for tech in tech_indicators:
            if tech in text:
                scraped_data['tech_stack'].append(tech)
        
        # API and developer signals
        if any(word in text for word in ['api documentation', 'developer guide', 'api reference']):
            scraped_data['api_documentation'] = True
        
        if any(word in text for word in ['developer portal', 'sandbox', 'test environment']):
            scraped_data['developer_resources'] = True
        
        # Employee count signals
        employee_patterns = [
            r'(\d+)\+?\s*employees',
            r'team of (\d+)',
            r'(\d+)\s*people',
            r'staff of (\d+)',
        ]
        
        for pattern in employee_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                scraped_data['employee_signals'].extend([int(m) for m in matches])
        
        # Funding signals
        funding_patterns = [
            r'series [abc]',
            r'seed funding',
            r'raised \$(\d+(?:\.\d+)?)[mk]?',
            r'funding of \$(\d+(?:\.\d+)?)[mk]?',
            r'valuation of \$(\d+(?:\.\d+)?)[bmk]?'
        ]
        
        for pattern in funding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                scraped_data['funding_signals'].extend(matches)

class APIEnricher:
    """External API enrichment component"""
    
    def __init__(self):
        self.clearbit_enabled = ENRICHMENT_SOURCES['clearbit']['enabled']
        self.clearbit_api_key = ENRICHMENT_SOURCES['clearbit']['api_key']
    
    async def enrich_with_clearbit(self, domain: str) -> Dict[str, Any]:
        """Enrich company data using Clearbit API"""
        if not self.clearbit_enabled:
            return {'error': 'Clearbit not enabled', 'confidence': 0.0}
        
        try:
            headers = {'Authorization': f'Bearer {self.clearbit_api_key}'}
            url = f"https://company-stream.clearbit.com/v2/companies/find?domain={domain}"
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'name': data.get('name'),
                            'industry': data.get('category', {}).get('industry'),
                            'employees': data.get('metrics', {}).get('employees'),
                            'revenue': data.get('metrics', {}).get('annualRevenue'),
                            'funding': data.get('metrics', {}).get('raised'),
                            'tech': data.get('tech', []),
                            'confidence': 0.9  # High confidence for API data
                        }
                    else:
                        return {'error': f'Clearbit API error: {response.status}', 'confidence': 0.0}
        
        except Exception as e:
            logger.error(f"Clearbit enrichment failed for {domain}: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def enrich_with_domain_analysis(self, domain: str) -> Dict[str, Any]:
        """Analyze domain for technical and business signals"""
        try:
            # DNS and domain analysis
            domain_data = {
                'domain_age': await self._estimate_domain_age(domain),
                'subdomains': await self._detect_subdomains(domain),
                'ssl_certificate': await self._check_ssl(domain),
                'cdn_usage': await self._detect_cdn(domain),
                'confidence': 0.7
            }
            
            return domain_data
            
        except Exception as e:
            logger.error(f"Domain analysis failed for {domain}: {e}")
            return {'error': str(e), 'confidence': 0.0}
    
    async def _estimate_domain_age(self, domain: str) -> Optional[str]:
        """Estimate domain age (simplified implementation)"""
        # In a real implementation, you'd use whois data
        return "unknown"
    
    async def _detect_subdomains(self, domain: str) -> List[str]:
        """Detect common subdomains"""
        common_subdomains = ['api', 'app', 'www', 'docs', 'developer', 'blog']
        found_subdomains = []
        
        for subdomain in common_subdomains:
            try:
                test_domain = f"https://{subdomain}.{domain}"
                async with aiohttp.ClientSession() as session:
                    async with session.head(test_domain, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            found_subdomains.append(subdomain)
            except:
                continue
        
        return found_subdomains
    
    async def _check_ssl(self, domain: str) -> bool:
        """Check if domain has valid SSL certificate"""
        try:
            url = f"https://{domain}"
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status < 400
        except:
            return False
    
    async def _detect_cdn(self, domain: str) -> Optional[str]:
        """Detect CDN usage"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(f"https://{domain}") as response:
                    headers = response.headers
                    
                    # Common CDN headers
                    if 'cloudflare' in str(headers).lower():
                        return 'cloudflare'
                    elif 'amazonaws' in str(headers).lower():
                        return 'aws'
                    elif 'fastly' in str(headers).lower():
                        return 'fastly'
                    
                    return None
        except:
            return None

class CompanyEnrichmentEngine:
    """Main orchestrator for company enrichment"""
    
    def __init__(self):
        self.web_scraper = None
        self.api_enricher = APIEnricher()
    
    async def enrich_company(self, name: str, domain: str) -> CompanyProfile:
        """Enrich a company with all available data sources"""
        logger.info(f"Enriching company: {name} ({domain})")
        
        profile = CompanyProfile(name=name, domain=domain)
        enrichment_tasks = []
        
        # Web scraping
        if ENRICHMENT_SOURCES['web_scraping']['enabled']:
            async with WebScraper() as scraper:
                web_data = await scraper.scrape_company_website(domain)
                profile.enrichment_sources.append('web_scraping')
                self._merge_web_data(profile, web_data)
        
        # API enrichment
        if ENRICHMENT_SOURCES['clearbit']['enabled']:
            clearbit_data = await self.api_enricher.enrich_with_clearbit(domain)
            profile.enrichment_sources.append('clearbit')
            self._merge_api_data(profile, clearbit_data)
        
        # Domain analysis
        if ENRICHMENT_SOURCES['domain_analysis']['enabled']:
            domain_data = await self.api_enricher.enrich_with_domain_analysis(domain)
            profile.enrichment_sources.append('domain_analysis')
            self._merge_domain_data(profile, domain_data)
        
        # Calculate overall confidence
        profile.enrichment_confidence = self._calculate_confidence(profile)
        
        logger.info(f"Enrichment complete for {name}: confidence {profile.enrichment_confidence:.2f}")
        return profile
    
    def _merge_web_data(self, profile: CompanyProfile, web_data: Dict[str, Any]):
        """Merge web scraping data into profile"""
        if 'error' in web_data:
            return
        
        # Tech signals
        profile.tech_signals.update({
            'product_mentions': web_data.get('product_mentions', {}),
            'use_case_signals': web_data.get('use_case_signals', {}),
            'tech_stack': web_data.get('tech_stack', []),
            'api_documentation': web_data.get('api_documentation', False),
            'developer_resources': web_data.get('developer_resources', False)
        })
        
        # Employee signals
        employee_signals = web_data.get('employee_signals', [])
        if employee_signals:
            profile.employees_count = max(employee_signals)  # Use highest mentioned number
    
    def _merge_api_data(self, profile: CompanyProfile, api_data: Dict[str, Any]):
        """Merge API enrichment data into profile"""
        if 'error' in api_data:
            return
        
        if api_data.get('name'):
            profile.name = api_data['name']
        if api_data.get('industry'):
            profile.industry = api_data['industry']
        if api_data.get('employees'):
            profile.employees_count = api_data['employees']
        if api_data.get('revenue'):
            profile.revenue_range = self._format_revenue(api_data['revenue'])
        if api_data.get('funding'):
            profile.funding_amount = api_data['funding']
        if api_data.get('tech'):
            profile.tech_signals['clearbit_tech'] = api_data['tech']
    
    def _merge_domain_data(self, profile: CompanyProfile, domain_data: Dict[str, Any]):
        """Merge domain analysis data into profile"""
        if 'error' in domain_data:
            return
        
        profile.tech_signals.update({
            'subdomains': domain_data.get('subdomains', []),
            'ssl_certificate': domain_data.get('ssl_certificate', False),
            'cdn_usage': domain_data.get('cdn_usage'),
            'domain_age': domain_data.get('domain_age')
        })
    
    def _format_revenue(self, revenue: int) -> str:
        """Format revenue into range"""
        if revenue < 1000000:
            return "Under $1M"
        elif revenue < 10000000:
            return "$1M-$10M"
        elif revenue < 100000000:
            return "$10M-$100M"
        else:
            return "$100M+"
    
    def _calculate_confidence(self, profile: CompanyProfile) -> float:
        """Calculate overall enrichment confidence"""
        confidence_factors = []
        
        # Data completeness
        if profile.industry:
            confidence_factors.append(0.2)
        if profile.employees_count:
            confidence_factors.append(0.2)
        if profile.tech_signals:
            confidence_factors.append(0.3)
        if len(profile.enrichment_sources) > 1:
            confidence_factors.append(0.2)
        if profile.tech_signals.get('api_documentation'):
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)

# Batch enrichment functions
class BatchEnrichmentProcessor:
    """Process multiple companies in batches"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.enrichment_engine = CompanyEnrichmentEngine()
    
    async def enrich_companies_batch(self, companies: List[Tuple[str, str]]) -> List[CompanyProfile]:
        """Enrich multiple companies concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def enrich_with_limit(name: str, domain: str):
            async with semaphore:
                return await self.enrichment_engine.enrich_company(name, domain)
        
        tasks = [enrich_with_limit(name, domain) for name, domain in companies]
        profiles = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_profiles = [p for p in profiles if isinstance(p, CompanyProfile)]
        logger.info(f"Successfully enriched {len(valid_profiles)}/{len(companies)} companies")
        
        return valid_profiles

# Example usage
async def main():
    """Test the enrichment engine"""
    companies_to_test = [
        ("Structurely", "structurely.com"),
        ("Telnyx", "telnyx.com"),
        ("OpenAI", "openai.com"),
    ]
    
    processor = BatchEnrichmentProcessor()
    profiles = await processor.enrich_companies_batch(companies_to_test)
    
    for profile in profiles:
        print(f"\nCompany: {profile.name}")
        print(f"Domain: {profile.domain}")
        print(f"Industry: {profile.industry}")
        print(f"Employees: {profile.employees_count}")
        print(f"Tech Signals: {len(profile.tech_signals)}")
        print(f"Confidence: {profile.enrichment_confidence:.2f}")
        print(f"Sources: {profile.enrichment_sources}")

if __name__ == "__main__":
    asyncio.run(main())