"""
Configuration settings for Fellow.ai Learning Qualification System
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
FELLOW_API_CONFIG = {
    "base_url": "https://api.fellow.app",
    "version": "v1",
    "timeout": 30,
    "rate_limit": 100,  # requests per minute
}

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "name": os.getenv("DB_NAME", "fellow_learning"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "xgboost",
    "retrain_threshold": 100,  # new samples before retraining
    "performance_threshold": 0.75,  # minimum accuracy
    "confidence_threshold": 0.7,  # minimum confidence for predictions
    "feature_importance_threshold": 0.01,
}

# Data Pipeline Configuration
PIPELINE_CONFIG = {
    "batch_size": 50,
    "max_concurrent_enrichments": 10,
    "enrichment_timeout": 300,  # seconds
    "cache_duration": 3600,  # seconds
    "retry_attempts": 3,
}

# Enrichment Sources
ENRICHMENT_SOURCES = {
    "clearbit": {
        "api_key": os.getenv("CLEARBIT_API_KEY"),
        "enabled": bool(os.getenv("CLEARBIT_API_KEY")),
    },
    "web_scraping": {
        "enabled": True,
        "timeout": 30,
        "max_pages": 5,
    },
    "domain_analysis": {
        "enabled": True,
        "tech_stack_detection": True,
    }
}

# NLP Configuration
NLP_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "overlap": 50,
    "min_confidence": 0.6,
}

# Scoring Configuration
SCORING_CONFIG = {
    "voice_ai_weight": 3.0,  # High weight for Voice AI prospects
    "enterprise_weight": 2.0,  # Weight for enterprise signals
    "progression_weight": 1.5,  # Weight for progression patterns
    "recency_decay": 0.95,  # Decay factor for older data
}

# Product Categories for Detection
PRODUCT_CATEGORIES = {
    "Voice AI": [
        "voice ai", "conversational ai", "voice assistant", "ai voice",
        "voice automation", "ai calling", "voice bot", "speech ai",
        "voice recognition", "ai phone", "voice agent", "smart voice"
    ],
    "Voice": [
        "voice calling", "phone calls", "sip trunking", "phone numbers",
        "voice solutions", "telephony", "voice communication", "phone system",
        "voice services", "calling platform", "voice network"
    ],
    "Messaging": [
        "sms", "text messaging", "messaging api", "messaging platform",
        "whatsapp business", "rcs", "mms", "chat api", "messaging service"
    ],
    "Verify": [
        "2fa", "two factor", "phone verification", "otp", "verification",
        "identity verification", "authentication", "verify phone"
    ],
    "Video": [
        "video calling", "video conferencing", "video api", "webrtc",
        "video communication", "video platform", "video chat"
    ],
    "Wireless": [
        "iot connectivity", "wireless", "mobile connectivity", "cellular",
        "iot sim", "wireless iot", "connected devices"
    ]
}

# Use Case Signals
USE_CASE_SIGNALS = {
    "high_volume": [
        "million users", "high volume", "scale", "enterprise",
        "millions of", "large scale", "massive", "high traffic"
    ],
    "technical": [
        "api", "integration", "webhook", "sdk", "developer",
        "technical", "programming", "code", "platform"
    ],
    "urgent": [
        "urgent", "asap", "immediately", "quick", "fast",
        "time sensitive", "deadline", "rush"
    ],
    "budget_confirmed": [
        "budget", "approved", "funding", "allocated", "price",
        "cost", "investment", "procurement"
    ]
}

# Progression Indicators
PROGRESSION_SIGNALS = {
    "positive": [
        "pricing", "quote", "proposal", "next steps", "technical call",
        "demo", "poc", "pilot", "trial", "implementation", "contract",
        "procurement", "legal review", "technical integration"
    ],
    "neutral": [
        "follow up", "more information", "discuss internally",
        "think about it", "next week", "circle back"
    ],
    "negative": [
        "not a fit", "not interested", "no budget", "not now",
        "different solution", "went with", "decided against"
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "fellow_learning.log"),
            "formatter": "standard",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "fellow_learning": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

def get_database_url() -> str:
    """Get database connection URL"""
    return f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['name']}"

def get_api_headers() -> Dict[str, str]:
    """Get API headers with authentication"""
    return {
        "Authorization": f"Bearer {os.getenv('FELLOW_API_KEY')}",
        "Content-Type": "application/json",
        "User-Agent": "Fellow-Learning-System/1.0"
    }