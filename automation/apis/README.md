# API Integration Layer

Integration layer for Fellow.ai API and enrichment data sources with authentication, rate limiting, and error handling.

## 🔌 Supported APIs

### Fellow.ai API
- **Base URL**: `https://api.fellow.app/v1`
- **Endpoint**: `https://telnyx.fellow.app/api/v1/recordings`
- **Authentication**: API Key (X-Api-Key header)
- **Rate Limit**: No explicit limit, but implement backoff
- **Use Case**: Meeting data, call transcripts, AE sentiment

### Clearbit Enrichment API
- **Base URL**: `https://company.clearbit.com/v2`
- **Authentication**: Bearer token
- **Rate Limit**: 600 calls/hour
- **Use Case**: Company data, employee count, revenue, technologies

### OpenFunnel API
- **Base URL**: `https://api.openfunnel.com/v1` (placeholder)
- **Authentication**: API Key
- **Rate Limit**: 100 calls/hour
- **Use Case**: Additional company intelligence

## 🛡️ Security & Authentication

### API Key Management
```bash
# Environment variables for secure key storage
export FELLOW_API_KEY="c2e66647b10bfbc93b85cc1b05b8bc519bc61d849a09f5ac8f767fbad927dcc4"
export CLEARBIT_API_KEY="sk_your_clearbit_key_here"
export OPENFUNNEL_API_KEY="your_openfunnel_key_here"
```

### Request Headers
```python
# Fellow API
headers = {
    'X-Api-Key': FELLOW_API_KEY,
    'Content-Type': 'application/json',
    'User-Agent': 'Telnyx-Fellow-Automation/1.0'
}

# Clearbit API  
headers = {
    'Authorization': f'Bearer {CLEARBIT_API_KEY}',
    'Content-Type': 'application/json'
}
```

## ⚡ Rate Limiting & Error Handling

### Rate Limiting Strategy
```python
# Rate limit tracking per service
rate_limits = {
    'fellow': {'calls': 0, 'reset_time': 0, 'max_per_hour': 1000},
    'clearbit': {'calls': 0, 'reset_time': 0, 'max_per_hour': 600},
    'openfunnel': {'calls': 0, 'reset_time': 0, 'max_per_hour': 100}
}

# Check before making request
if not check_rate_limit('clearbit'):
    wait_time = calculate_wait_time('clearbit')
    time.sleep(wait_time)
```

### Error Recovery
```python
# Exponential backoff for failed requests
def make_api_request(url, headers, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
                time.sleep(wait_time)
            else:
                raise APIError(f"Failed after {max_retries} attempts: {e}")
```

## 📊 API Response Handling

### Fellow API Response Format
```json
{
  "recordings": [
    {
      "id": "meeting_001",
      "title": "Telnyx Intro Call - Company Name",
      "company_name": "Extracted Company",
      "date": "2026-02-03",
      "ae_name": "AE Name",
      "notes": "Meeting notes and content",
      "action_items_count": 4,
      "follow_up_scheduled": true
    }
  ]
}
```

### Clearbit API Response Format
```json
{
  "domain": "company.com",
  "name": "Company Name",
  "description": "Company description",
  "category": {
    "industry": "Software"
  },
  "metrics": {
    "employees": 150,
    "employeesRange": "100-500",
    "annualRevenue": 10000000
  },
  "tech": [
    {"name": "React"},
    {"name": "Node.js"}
  ]
}
```

## 🎯 API Client Examples

### Fellow API Client
```python
class FellowAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://telnyx.fellow.app/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Api-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def get_meetings(self, date_range, meeting_filter="Telnyx Intro Call"):
        params = {
            'date_range': date_range,
            'meeting_title': meeting_filter
        }
        response = self.session.get(f"{self.base_url}/recordings", params=params)
        response.raise_for_status()
        return response.json()
```

### Clearbit API Client  
```python
class ClearbitAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://company.clearbit.com/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def find_company(self, domain=None, name=None):
        params = {}
        if domain:
            params['domain'] = domain
        elif name:
            params['name'] = name
        else:
            raise ValueError("Either domain or name required")
            
        response = self.session.get(f"{self.base_url}/companies/find", params=params)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
```

---

**Note**: This API integration layer is designed to be modular and extensible. New data sources can be easily added by implementing the same patterns for authentication, rate limiting, and error handling.