# Fellow Learning Qualification System Setup Guide

## Prerequisites

### System Requirements
- **Python 3.9+** with pip and virtualenv
- **PostgreSQL 12+** with database creation privileges
- **Redis 6+** for task queuing and caching
- **Git** for version control and deployment

### API Access Required
- **Fellow.ai API Key** - Contact Fellow.ai support for access
- **Optional Enrichment APIs:**
  - Clearbit API (recommended for enhanced company data)
  - ZoomInfo API (enterprise company intelligence)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/fellow-learning-qualification-system.git
cd fellow-learning-qualification-system
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup
```bash
# Install PostgreSQL (if not already installed)
# macOS with Homebrew:
brew install postgresql
brew services start postgresql

# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# Create database and user
psql postgres
CREATE DATABASE fellow_learning;
CREATE USER fellow_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE fellow_learning TO fellow_user;
\q

# Initialize database schema
python -c "
import sys
sys.path.append('.')
from config.database_schema import initialize_database
initialize_database()
"
```

### 4. Redis Setup
```bash
# Install Redis (if not already installed)
# macOS with Homebrew:
brew install redis
brew services start redis

# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis-server

# Test Redis connection
redis-cli ping
# Should return: PONG
```

### 5. Environment Configuration
```bash
# Copy environment template
cp config/.env.example config/.env

# Edit configuration with your values
nano config/.env
```

**Required Environment Variables:**
```bash
# Fellow API
FELLOW_API_KEY=your_fellow_api_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fellow_learning
DB_USER=fellow_user
DB_PASSWORD=your_password

# Redis
REDIS_URL=redis://localhost:6379/0

# Optional APIs
CLEARBIT_API_KEY=your_clearbit_key_here
```

### 6. Download NLP Models
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download sentence transformer model (will auto-download on first use)
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✓ Sentence transformer model downloaded')
"
```

## Testing Installation

### 1. Database Connection Test
```bash
python -c "
import os
import sys
sys.path.append('.')
from config.settings import get_database_url
import psycopg2

try:
    conn = psycopg2.connect(get_database_url())
    print('✓ Database connection successful')
    conn.close()
except Exception as e:
    print(f'✗ Database connection failed: {e}')
"
```

### 2. API Integration Test
```bash
python -c "
import os
import sys
import asyncio
sys.path.append('.')
from api.fellow_client import FellowAPIClient

async def test_fellow_api():
    try:
        async with FellowAPIClient() as client:
            connected = await client.test_connection()
            if connected:
                print('✓ Fellow API connection successful')
            else:
                print('✗ Fellow API connection failed')
    except Exception as e:
        print(f'✗ Fellow API error: {e}')

asyncio.run(test_fellow_api())
"
```

### 3. NLP Processing Test
```bash
python -c "
import sys
sys.path.append('.')
from analysis.call_analyzer import CallAnalysisEngine

try:
    analyzer = CallAnalysisEngine()
    test_result = analyzer.analyze_call('test-001', 'This is a test transcript about Voice AI solutions.')
    print(f'✓ NLP processing successful: {test_result.overall_score}')
except Exception as e:
    print(f'✗ NLP processing failed: {e}')
"
```

### 4. Complete Integration Test
```bash
python tests/test_integration.py
```

## Running the System

### 1. Start Background Services
```bash
# Terminal 1 - Start Redis (if not running as service)
redis-server

# Terminal 2 - Start Celery worker for background tasks
celery -A automation.celery_app worker --loglevel=info

# Terminal 3 - Start Celery beat for scheduled tasks
celery -A automation.celery_app beat --loglevel=info
```

### 2. Start API Server
```bash
# Development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production server (with gunicorn)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Start Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```

### 4. Manual Data Pipeline Test
```bash
# Fetch recent Fellow calls
python automation/data_pipeline.py fetch-calls --days 7

# Run enrichment on specific company
python automation/data_pipeline.py enrich-company --domain example.com

# Analyze specific call
python automation/data_pipeline.py analyze-call --call-id fellow-call-123

# Train model with current data
python ml-model/train_model.py --retrain
```

## Development Setup

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Pre-commit Hooks
```bash
pre-commit install
```

### 3. Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Full test suite with coverage
pytest --cov=. --cov-report=html
```

### 4. Code Quality Checks
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Production Deployment

### 1. Docker Deployment
```bash
# Build Docker image
docker build -t fellow-learning-system .

# Run with Docker Compose
docker-compose up -d
```

### 2. Environment Setup
```bash
# Production environment file
cp config/.env.example config/.env.production

# Edit with production values
nano config/.env.production
```

### 3. Database Migration
```bash
# Run database migrations
alembic upgrade head

# Create initial admin user
python scripts/create_admin_user.py
```

### 4. Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database health check
curl http://localhost:8000/health/database

# Model health check
curl http://localhost:8000/health/model
```

## Monitoring & Maintenance

### 1. Log Monitoring
```bash
# View application logs
tail -f logs/fellow_learning.log

# View API logs
tail -f logs/api.log

# View Celery logs
tail -f logs/celery.log
```

### 2. Performance Monitoring
- **API Dashboard:** http://localhost:8000/docs (FastAPI Swagger UI)
- **System Dashboard:** http://localhost:8501 (Streamlit dashboard)
- **Database Monitoring:** Monitor PostgreSQL performance and query times

### 3. Model Performance
- **Accuracy Tracking:** Monitor prediction accuracy vs actual outcomes
- **Drift Detection:** Alert when model performance degrades
- **Retraining:** Automatic retraining when new data threshold reached

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check PostgreSQL service
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql      # Linux

# Check database exists
psql postgres -c "\l" | grep fellow_learning
```

#### 2. Fellow API Authentication
- Verify API key in config/.env
- Check API key permissions with Fellow.ai support
- Test with minimal API call first

#### 3. NLP Model Loading Issues
```bash
# Reinstall spaCy model
python -m spacy download en_core_web_sm --force

# Clear model cache
rm -rf ~/.cache/huggingface/
```

#### 4. Memory Issues
- Reduce batch sizes in config
- Increase system RAM or use swap
- Consider model size alternatives

### Support Resources

- **GitHub Issues:** Report bugs and request features
- **Documentation:** Comprehensive guides in /docs/
- **Architecture Reviews:** System design and integration help

## Next Steps

After successful installation:

1. **Configure Fellow API** - Set up daily data fetching schedule
2. **Train Initial Model** - Use historical data for baseline model
3. **Set Up Monitoring** - Configure alerts and performance tracking  
4. **Integrate with Existing Systems** - Connect to current lead routing
5. **Begin Production Testing** - A/B test enhanced vs baseline qualification

---

**🎯 Installation Complete!** Your Fellow Learning Qualification System is ready for development and testing.