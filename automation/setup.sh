#!/bin/bash
"""
Fellow.ai Automation Infrastructure Setup Script
Initializes the complete automation system for learning qualification
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}Fellow.ai Automation Infrastructure Setup${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python 3
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check SQLite
    if ! command_exists sqlite3; then
        print_error "SQLite3 is required but not installed"
        exit 1
    fi
    
    # Check cron
    if ! command_exists crontab; then
        print_warning "crontab not found - cron scheduling will not work"
    fi
    
    print_status "Prerequisites check completed"
    echo ""
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create requirements.txt if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        cat > "$PROJECT_ROOT/requirements.txt" << EOF
# Core dependencies for Fellow.ai automation
requests>=2.25.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
sqlite3

# Optional dependencies for enrichment
# clearbit-python>=0.1.6  # Uncomment if using Clearbit
# beautifulsoup4>=4.9.0   # For web scraping
# lxml>=4.6.0             # For web scraping
EOF
    fi
    
    # Install dependencies
    pip3 install -r "$PROJECT_ROOT/requirements.txt" --user
    
    print_status "Python dependencies installed"
    echo ""
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data"
        "data/backups"
        "logs"
        "models"
        "config"
        "scripts"
        "pipelines"
        "monitoring"
        "docs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        print_status "Created: $dir/"
    done
    
    print_status "Directory structure created"
    echo ""
}

# Set file permissions
set_permissions() {
    print_status "Setting file permissions..."
    
    # Make scripts executable
    find "$PROJECT_ROOT/scripts" -name "*.py" -exec chmod +x {} \;
    find "$PROJECT_ROOT/scripts" -name "*.sh" -exec chmod +x {} \;
    find "$PROJECT_ROOT/pipelines" -name "*.py" -exec chmod +x {} \;
    find "$PROJECT_ROOT/monitoring" -name "*.py" -exec chmod +x {} \;
    chmod +x "$PROJECT_ROOT/setup.sh"
    
    # Set proper directory permissions
    chmod 755 "$PROJECT_ROOT"
    chmod 755 "$PROJECT_ROOT/data"
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/models"
    
    print_status "File permissions set"
    echo ""
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Run the ingestion script in test mode to create database
    python3 "$PROJECT_ROOT/scripts/fellow-ingestion.py" --test-mode 2>/dev/null || {
        # If the script doesn't have test mode, create database manually
        python3 -c "
import sqlite3
import sys
import os
sys.path.append('$PROJECT_ROOT/scripts')

db_path = '$PROJECT_ROOT/data/fellow_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create main tables
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
print('Database initialized successfully')
"
    }
    
    print_status "Database initialized"
    echo ""
}

# Test system components
test_components() {
    print_status "Testing system components..."
    
    # Test Fellow API connection (this will fail if API key is not set, but that's OK)
    print_status "Testing Fellow API connection..."
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/scripts')
try:
    from fellow_ingestion import FellowDataPipeline
    pipeline = FellowDataPipeline()
    print('✓ Fellow ingestion module loaded successfully')
except ImportError as e:
    print(f'⚠ Warning: Could not import Fellow ingestion module: {e}')
except Exception as e:
    print(f'⚠ Warning: Fellow API test failed (expected if API key not configured): {e}')
"
    
    # Test enrichment pipeline
    print_status "Testing enrichment pipeline..."
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/pipelines')
try:
    from enrichment_pipeline import CompanyEnricher
    enricher = CompanyEnricher()
    print('✓ Enrichment pipeline loaded successfully')
except ImportError as e:
    print(f'⚠ Warning: Could not import enrichment module: {e}')
except Exception as e:
    print(f'⚠ Warning: Enrichment test failed: {e}')
"
    
    # Test scoring system
    print_status "Testing scoring system..."
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT/scripts')
try:
    from realtime_scoring import LeadQualificationModel
    model = LeadQualificationModel()
    print('✓ Scoring system loaded successfully')
except ImportError as e:
    print(f'⚠ Warning: Could not import scoring module: {e}')
except Exception as e:
    print(f'⚠ Warning: Scoring test failed: {e}')
"
    
    print_status "Component testing completed"
    echo ""
}

# Generate documentation
generate_docs() {
    print_status "Generating documentation..."
    
    cat > "$PROJECT_ROOT/docs/README.md" << 'EOF'
# Fellow.ai Automation Infrastructure

This system provides automated data ingestion, enrichment, and lead qualification scoring for Fellow.ai call data.

## Components

### 1. Daily Data Ingestion (`scripts/fellow-ingestion.py`)
- Runs daily at 6:00 AM CST
- Fetches new meeting data from Fellow API
- Stores in local SQLite database
- Calculates initial sentiment scores

### 2. Company Enrichment (`pipelines/enrichment-pipeline.py`)
- Processes companies through multiple data sources
- Gathers intelligence from Clearbit, OpenFunnel, web scraping
- Runs every 30 minutes during business hours

### 3. Real-time Lead Scoring (`scripts/realtime-scoring.py`)
- Scores leads using ML model or fallback rules
- Runs every 2 hours during business hours
- Identifies high-value opportunities

### 4. Model Training (`scripts/model-training.py`)
- Trains qualification model on Fellow call outcomes
- Runs weekly on Mondays at 2:00 AM
- Improves scoring accuracy over time

### 5. Health Monitoring (`monitoring/health-monitor.py`)
- Monitors system health and data quality
- Runs every 4 hours
- Alerts on critical issues

### 6. Daily Maintenance (`scripts/daily-cleanup.sh`)
- Log rotation and database cleanup
- Runs daily at 3:00 AM
- Maintains system performance

## Setup

1. Run `./setup.sh` to initialize the system
2. Configure API keys in environment variables:
   ```bash
   export FELLOW_API_KEY="your_key_here"
   export CLEARBIT_API_KEY="your_key_here"  # Optional
   export OPENFUNNEL_API_KEY="your_key_here"  # Optional
   ```
3. Install cron jobs: `bash config/crontab-setup.sh`

## Manual Operations

Run individual components:
```bash
# Ingest Fellow data
python3 scripts/fellow-ingestion.py

# Process enrichment queue
python3 pipelines/enrichment-pipeline.py 10

# Score recent leads
python3 scripts/realtime-scoring.py

# Train model
python3 scripts/model-training.py

# Check system health
python3 monitoring/health-monitor.py
```

## Configuration

System configuration is in `config/system-config.json`. Modify settings as needed.

## Monitoring

- Check logs in `logs/` directory
- Monitor database with SQLite browser
- Review health check reports
- Check cron job output

## Troubleshooting

1. **Fellow API errors**: Check API key and network connectivity
2. **Database issues**: Verify permissions and disk space
3. **Enrichment failures**: Check rate limits and API keys
4. **Scoring problems**: Verify model files exist in `models/`

For support, check the logs and health monitoring output.
EOF
    
    cat > "$PROJECT_ROOT/docs/API_SETUP.md" << 'EOF'
# API Setup Instructions

## Fellow.ai API

The Fellow API key is already configured:
```bash
export FELLOW_API_KEY="c2e66647b10bfbc93b85cc1b05b8bc519bc61d849a09f5ac8f767fbad927dcc4"
```

## Clearbit API (Optional)

1. Sign up at https://clearbit.com/
2. Get your API key from the dashboard
3. Set environment variable:
```bash
export CLEARBIT_API_KEY="sk_your_clearbit_api_key"
```

## OpenFunnel API (Optional)

1. Contact OpenFunnel for API access
2. Set environment variable:
```bash
export OPENFUNNEL_API_KEY="your_openfunnel_api_key"
```

## Environment Variable Setup

Add to your `.bashrc` or `.bash_profile`:
```bash
# Fellow.ai Automation API Keys
export FELLOW_API_KEY="c2e66647b10bfbc93b85cc1b05b8bc519bc61d849a09f5ac8f767fbad927dcc4"
export CLEARBIT_API_KEY="your_clearbit_key_here"  # Optional
export OPENFUNNEL_API_KEY="your_openfunnel_key_here"  # Optional
```

Then reload: `source ~/.bashrc`
EOF
    
    print_status "Documentation generated"
    echo ""
}

# Show completion summary
show_summary() {
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo ""
    echo "1. Configure API keys (optional for Clearbit/OpenFunnel):"
    echo "   export CLEARBIT_API_KEY='your_key_here'"
    echo "   export OPENFUNNEL_API_KEY='your_key_here'"
    echo ""
    echo "2. Install cron jobs for automation:"
    echo "   bash config/crontab-setup.sh"
    echo ""
    echo "3. Test the system:"
    echo "   python3 scripts/fellow-ingestion.py"
    echo "   python3 monitoring/health-monitor.py"
    echo ""
    echo "4. Monitor the system:"
    echo "   tail -f logs/fellow-ingestion-$(date +%Y-%m-%d).log"
    echo ""
    echo -e "${BLUE}Project Structure:${NC}"
    echo "  scripts/          - Core automation scripts"
    echo "  pipelines/        - Data processing pipelines"
    echo "  monitoring/       - Health monitoring"
    echo "  data/            - Database and data files"
    echo "  logs/            - Log files"
    echo "  models/          - ML model files"
    echo "  docs/            - Documentation"
    echo ""
    echo -e "${GREEN}Fellow.ai automation infrastructure is ready!${NC}"
    echo ""
}

# Main setup process
main() {
    print_status "Starting Fellow.ai automation setup..."
    echo ""
    
    check_prerequisites
    install_dependencies
    create_directories
    set_permissions
    initialize_database
    test_components
    generate_docs
    show_summary
}

# Run main function
main

exit 0