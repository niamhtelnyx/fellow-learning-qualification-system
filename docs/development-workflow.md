# Development Workflow & Collaboration

## GitHub Repository Structure

**Repository URL:** https://github.com/niamhtelnyx/fellow-ai-learning-system

### Branch Strategy
- **`main`** - Production-ready code with complete features
- **`develop`** - Integration branch for ongoing development
- **`feature/*`** - Individual feature development branches
- **`hotfix/*`** - Critical bug fixes for production

### Subagent Development Areas

#### 🏗️ System Architect (This Subagent)
**Responsibilities:**
- Overall system architecture and design
- API integrations (Fellow, enrichment sources)
- Call analysis and NLP processing
- Database schema and data models
- Documentation and specifications

**Primary Directories:**
- `/architecture/` - System designs and specifications  
- `/api/` - API integration code
- `/analysis/` - Call analysis and NLP processing
- `/config/` - Configuration and database schema
- `/docs/` - Documentation and guides

#### 🔄 Automation Engineer  
**Responsibilities:**
- Daily data pipeline automation
- Batch processing and job scheduling
- Data quality monitoring and validation
- ETL processes and data transformation

**Primary Directories:**
- `/automation/` - Cron jobs, schedulers, pipelines
- `/scripts/` - Data processing and maintenance scripts
- `/tests/integration/` - Pipeline integration testing

#### 🤖 ML Engineer
**Responsibilities:**
- Feature engineering and model development
- Model training, evaluation, and optimization  
- Real-time prediction API development
- Performance monitoring and model drift detection

**Primary Directories:**
- `/ml-model/` - Model code, training, inference
- `/scoring/` - Real-time prediction API
- `/tests/ml/` - Model testing and validation

## Commit Standards

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- **feat:** New feature development
- **fix:** Bug fixes  
- **docs:** Documentation updates
- **style:** Code formatting, no logic changes
- **refactor:** Code restructuring without feature changes
- **test:** Adding or updating tests
- **chore:** Maintenance tasks, dependency updates

### Examples
```bash
feat(api): Add Fellow API rate limiting and retry logic

Implements exponential backoff retry mechanism and respects
API rate limits to prevent service disruptions.

- Add RateLimiter class with configurable limits
- Implement async retry with backoff
- Add comprehensive error handling
- Update tests for new retry behavior

Closes #15
```

```bash
fix(enrichment): Handle missing domain gracefully

Web scraper now validates domains before processing and
skips enrichment for invalid/missing domains.

- Add domain validation in WebScraper
- Return empty result instead of throwing exception  
- Add logging for skipped domains
- Update confidence scoring for partial data

Fixes #23
```

### Scope Guidelines
- **api:** Fellow API, external APIs, HTTP clients
- **enrichment:** Company enrichment, web scraping
- **analysis:** Call analysis, NLP processing
- **ml:** Machine learning, models, training
- **automation:** Data pipelines, scheduled jobs
- **config:** Configuration, environment, database
- **docs:** Documentation, README updates
- **tests:** Test additions, updates, fixes

## Development Process

### 1. Feature Development
```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/voice-ai-detection

# Develop feature with regular commits
git add .
git commit -m "feat(analysis): Add Voice AI keyword detection"

# Push branch and create pull request
git push origin feature/voice-ai-detection
```

### 2. Pull Request Process
1. **Create PR** from feature branch to `develop`
2. **Add Description** with feature overview and testing notes
3. **Request Review** from relevant subagent(s)
4. **Run Tests** and ensure CI passes
5. **Address Feedback** and update as needed
6. **Merge** when approved and tests pass

### 3. Integration Testing
```bash
# Checkout develop branch
git checkout develop
git pull origin develop

# Run full test suite
pytest tests/ --cov=. --cov-report=html

# Test integration with other components
pytest tests/integration/ -v

# Performance testing
python tests/performance/test_pipeline.py
```

### 4. Release Process
```bash
# Create release branch from develop
git checkout develop
git checkout -b release/v1.1.0

# Update version numbers and documentation
# Run final testing and bug fixes

# Merge to main and tag release
git checkout main
git merge release/v1.1.0
git tag -a v1.1.0 -m "Release v1.1.0: Enhanced Voice AI detection"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge main
git push origin develop
```

## Code Quality Standards

### 1. Python Code Style
- **Formatter:** Black with line length 88
- **Linter:** Flake8 with standard configuration
- **Type Hints:** mypy for static type checking
- **Import Sorting:** isort with Black-compatible settings

### 2. Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Hooks run automatically on commit:
# - Black formatting
# - Flake8 linting  
# - mypy type checking
# - isort import sorting
# - Trailing whitespace removal
```

### 3. Testing Requirements
- **Unit Tests:** pytest for all new functions and classes
- **Integration Tests:** End-to-end testing of major workflows
- **Coverage:** Minimum 80% code coverage for new code
- **Performance Tests:** Benchmark critical paths

### 4. Documentation Standards
- **Docstrings:** Google-style docstrings for all public functions
- **Type Hints:** All function parameters and returns
- **README Updates:** Keep project status and setup current
- **Architecture Docs:** Update specs for major changes

## Collaboration Guidelines

### 1. Communication
- **GitHub Issues:** Track bugs, features, and tasks
- **Pull Request Reviews:** Collaborative code review
- **Architecture Discussions:** Major design decisions in issues
- **Status Updates:** Regular progress commits and documentation

### 2. Integration Points
```python
# Example: Automation Engineer using Architect's API
from api.fellow_client import FellowDataFetcher
from analysis.call_analyzer import BatchCallAnalyzer

# Clear interface for pipeline integration
fetcher = FellowDataFetcher()
analyzer = BatchCallAnalyzer()

calls = await fetcher.fetch_daily_calls()
results = await analyzer.analyze_calls_batch(calls)
```

### 3. Shared Resources
- **Database Schema:** Coordinate changes through migrations
- **Configuration:** Use shared settings.py for consistency
- **Test Data:** Shared fixtures and sample data
- **Documentation:** Update shared docs for interface changes

## Issue Tracking

### 1. Issue Labels
- **`bug`** - Something isn't working
- **`enhancement`** - New feature or improvement
- **`documentation`** - Documentation updates needed
- **`good first issue`** - Good for newcomers
- **`help wanted`** - Extra attention needed
- **`priority:high`** - Critical issues
- **`area:api`** - API-related issues
- **`area:ml`** - Machine learning issues
- **`area:automation`** - Pipeline and automation issues

### 2. Issue Templates
**Bug Report:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g. macOS, Ubuntu]
- Python version: [e.g. 3.9.7]
- Component: [e.g. enrichment, api]
```

**Feature Request:**
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Additional context**
Any other context about the feature request.
```

## Performance Monitoring

### 1. Key Metrics
- **API Response Times:** Track Fellow API and enrichment performance
- **Data Processing Throughput:** Calls processed per hour
- **Model Accuracy:** Prediction accuracy over time
- **System Resources:** Memory, CPU, disk usage

### 2. Monitoring Tools
- **Application Metrics:** Custom metrics in code
- **Database Performance:** PostgreSQL query monitoring
- **API Health Checks:** Endpoint monitoring
- **Model Drift Detection:** Automated accuracy tracking

### 3. Alerting Thresholds
```python
# Example monitoring configuration
MONITORING_CONFIG = {
    'api_response_time_ms': 1000,  # Alert if >1 second
    'model_accuracy_threshold': 0.75,  # Alert if <75%
    'data_processing_delay_hours': 2,  # Alert if >2 hours behind
    'disk_usage_threshold': 0.85,  # Alert if >85% full
}
```

---

**🚀 Development Repository:** https://github.com/niamhtelnyx/fellow-ai-learning-system

**📊 Status:** Foundation complete, ready for collaborative development