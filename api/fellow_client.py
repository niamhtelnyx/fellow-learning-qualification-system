"""
Fellow.ai API Client for Learning Qualification System
Handles authentication, rate limiting, and data fetching from Fellow API
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import time
from pathlib import Path
import json

from ..config.settings import FELLOW_API_CONFIG, get_api_headers

logger = logging.getLogger('fellow_learning.api')

@dataclass
class FellowCall:
    """Represents a Fellow.ai call record"""
    fellow_id: str
    title: str
    date_time: datetime
    duration: Optional[int]  # in minutes
    participants: List[Dict[str, Any]]
    transcript: Optional[str]
    summary: Optional[str]
    outcome: Optional[str]
    raw_data: Dict[str, Any]

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        if len(self.requests) >= self.max_requests:
            # Wait until we can make another request
            wait_time = self.time_window - (now - self.requests[0]) + 1
            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        self.requests.append(now)

class FellowAPIClient:
    """Client for Fellow.ai API with rate limiting and error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = FELLOW_API_CONFIG["base_url"]
        self.api_key = api_key
        self.rate_limiter = RateLimiter(FELLOW_API_CONFIG["rate_limit"])
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=FELLOW_API_CONFIG["timeout"]),
            headers=get_api_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make a rate-limited API request"""
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/{FELLOW_API_CONFIG['version']}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Successfully fetched {endpoint}")
                    return data
                elif response.status == 401:
                    logger.error("API authentication failed - check API key")
                    raise Exception("Authentication failed")
                elif response.status == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return await self._make_request(endpoint, params)
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    response.raise_for_status()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error during API request: {e}")
            raise
    
    async def get_meetings(self, 
                          start_date: datetime = None, 
                          end_date: datetime = None,
                          limit: int = 100) -> List[FellowCall]:
        """
        Fetch meetings from Fellow API
        
        Args:
            start_date: Fetch meetings from this date onwards
            end_date: Fetch meetings until this date
            limit: Maximum number of meetings to fetch per request
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)  # Default to last week
        
        if end_date is None:
            end_date = datetime.now()
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": limit,
            "include": "transcript,participants,summary"
        }
        
        all_calls = []
        offset = 0
        
        while True:
            params["offset"] = offset
            
            try:
                response = await self._make_request("meetings", params)
                meetings = response.get("meetings", [])
                
                if not meetings:
                    break
                
                # Convert to FellowCall objects
                for meeting_data in meetings:
                    try:
                        call = self._parse_meeting(meeting_data)
                        if call and self._is_intro_call(call):
                            all_calls.append(call)
                    except Exception as e:
                        logger.warning(f"Failed to parse meeting {meeting_data.get('id', 'unknown')}: {e}")
                        continue
                
                logger.info(f"Fetched {len(meetings)} meetings (offset: {offset})")
                
                # Check if we have more data to fetch
                if len(meetings) < limit:
                    break
                    
                offset += limit
                
                # Prevent infinite loops
                if offset > 10000:
                    logger.warning("Reached maximum offset limit")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to fetch meetings at offset {offset}: {e}")
                break
        
        logger.info(f"Total intro calls fetched: {len(all_calls)}")
        return all_calls
    
    def _parse_meeting(self, meeting_data: Dict[str, Any]) -> Optional[FellowCall]:
        """Parse meeting data from Fellow API response"""
        try:
            # Parse datetime
            date_str = meeting_data.get("start_time") or meeting_data.get("created_at")
            if not date_str:
                logger.warning(f"No date found for meeting {meeting_data.get('id')}")
                return None
            
            date_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Extract participants
            participants = []
            for participant in meeting_data.get("participants", []):
                participants.append({
                    "name": participant.get("name"),
                    "email": participant.get("email"),
                    "role": participant.get("role"),
                    "company": participant.get("company"),
                    "external": participant.get("external", True)
                })
            
            # Get transcript
            transcript = None
            if "transcript" in meeting_data:
                if isinstance(meeting_data["transcript"], str):
                    transcript = meeting_data["transcript"]
                elif isinstance(meeting_data["transcript"], dict):
                    transcript = meeting_data["transcript"].get("content")
            
            return FellowCall(
                fellow_id=meeting_data["id"],
                title=meeting_data.get("title", ""),
                date_time=date_time,
                duration=meeting_data.get("duration"),  # in minutes
                participants=participants,
                transcript=transcript,
                summary=meeting_data.get("summary"),
                outcome=meeting_data.get("outcome"),
                raw_data=meeting_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing meeting data: {e}")
            return None
    
    def _is_intro_call(self, call: FellowCall) -> bool:
        """
        Determine if a call is an intro/discovery call based on title and participants
        """
        if not call.title:
            return False
        
        title_lower = call.title.lower()
        
        # Keywords that indicate intro/discovery calls
        intro_keywords = [
            "intro", "introduction", "discovery", "initial", "first call",
            "initial call", "kick-off", "kickoff", "meet and greet",
            "getting to know", "overview call", "exploratory",
            "qualification", "needs assessment", "demo request"
        ]
        
        # Check for intro keywords in title
        has_intro_keywords = any(keyword in title_lower for keyword in intro_keywords)
        
        # Check if there are external participants (prospects)
        has_external_participants = any(p.get("external", True) for p in call.participants)
        
        # Additional heuristics
        is_short_call = call.duration and call.duration <= 60  # Less than 1 hour
        
        # Return True if it looks like an intro call
        return has_intro_keywords and has_external_participants
    
    async def get_meeting_details(self, meeting_id: str) -> Optional[FellowCall]:
        """Get detailed information for a specific meeting"""
        try:
            response = await self._make_request(f"meetings/{meeting_id}")
            return self._parse_meeting(response)
        except Exception as e:
            logger.error(f"Failed to get meeting details for {meeting_id}: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test API connection and authentication"""
        try:
            await self._make_request("user/profile")
            logger.info("Fellow API connection successful")
            return True
        except Exception as e:
            logger.error(f"Fellow API connection failed: {e}")
            return False

class FellowDataFetcher:
    """High-level interface for fetching Fellow data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    async def fetch_daily_calls(self, date: datetime = None) -> List[FellowCall]:
        """Fetch calls for a specific day (default: yesterday)"""
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        logger.info(f"Fetching calls for {start_date.date()}")
        
        async with FellowAPIClient(self.api_key) as client:
            calls = await client.get_meetings(start_date=start_date, end_date=end_date)
        
        # Cache the results
        cache_file = self.cache_dir / f"calls_{start_date.date()}.json"
        with open(cache_file, 'w') as f:
            json.dump([{
                'fellow_id': call.fellow_id,
                'title': call.title,
                'date_time': call.date_time.isoformat(),
                'duration': call.duration,
                'participants': call.participants,
                'transcript': call.transcript,
                'summary': call.summary,
                'outcome': call.outcome,
                'raw_data': call.raw_data
            } for call in calls], f, indent=2)
        
        return calls
    
    async def fetch_historical_calls(self, days_back: int = 30) -> List[FellowCall]:
        """Fetch historical calls for model training"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching historical calls from {start_date.date()} to {end_date.date()}")
        
        async with FellowAPIClient(self.api_key) as client:
            calls = await client.get_meetings(start_date=start_date, end_date=end_date)
        
        return calls

# Example usage and testing
async def main():
    """Test the Fellow API client"""
    import os
    
    api_key = os.getenv("FELLOW_API_KEY")
    if not api_key:
        print("Please set FELLOW_API_KEY environment variable")
        return
    
    # Test connection
    async with FellowAPIClient(api_key) as client:
        connected = await client.test_connection()
        if not connected:
            print("Failed to connect to Fellow API")
            return
        
        # Fetch recent calls
        calls = await client.get_meetings(limit=5)
        print(f"Fetched {len(calls)} intro calls:")
        
        for call in calls:
            print(f"- {call.title} on {call.date_time.date()}")
            print(f"  Participants: {len(call.participants)}")
            if call.transcript:
                print(f"  Transcript: {len(call.transcript)} characters")

if __name__ == "__main__":
    asyncio.run(main())