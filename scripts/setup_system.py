#!/usr/bin/env python3
"""
Fellow Learning System Setup and Integration Script
Sets up the complete ML qualification system with sample data and testing
"""

import os
import sys
import json
import sqlite3
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "ml-model"))
sys.path.append(str(PROJECT_ROOT / "api"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directory_structure():
    """Create necessary directories"""
    logger.info("Setting up directory structure...")
    
    directories = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "logs", 
        PROJECT_ROOT / "ml-model" / "models",
        PROJECT_ROOT / "api",
        PROJECT_ROOT / "dashboard",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "config"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_requirements():
    """Install required Python packages"""
    logger.info("Installing Python requirements...")
    
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "streamlit>=1.12.0",
        "plotly>=5.0.0",
        "requests>=2.25.0",
        "joblib>=1.1.0",
        "pydantic>=1.8.0",
        "sqlite3"  # Usually comes with Python
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True, text=True)
            logger.info(f"Installed: {req}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {req}: {e}")

def create_sample_fellow_database():
    """Create sample Fellow database with test data"""
    logger.info("Creating sample Fellow database...")
    
    db_path = PROJECT_ROOT / "data" / "fellow_data.db"
    
    # Sample call data based on enrichment results
    sample_calls = [
        {
            'id': 'call_001',
            'title': 'Telnyx Intro Call - Structurely',
            'company_name': 'Structurely',
            'date': '2024-02-01',
            'ae_name': 'John Doe',
            'notes': 'Voice AI company with real estate focus. Currently using Twilio but experiencing reliability issues. Need high-volume calling for lead qualification. Discussed API integration and custom voice flows. Ready to move forward with POC.',
            'action_items_count': 3,
            'follow_up_scheduled': 1,
            'sentiment_score': 9,
            'strategic_score': 8,
            'processed': 1,
            'enriched': 1
        },
        {
            'id': 'call_002',
            'title': 'Telnyx Intro Call - UnifyGTM',
            'company_name': 'UnifyGTM',
            'date': '2024-02-02',
            'ae_name': 'Jane Smith',
            'notes': 'B2B sales automation platform using AI agents for calling sequences. Need reliable voice infrastructure for their AI calling product. Discussed enterprise pricing and compliance requirements.',
            'action_items_count': 2,
            'follow_up_scheduled': 1,
            'sentiment_score': 8,
            'strategic_score': 9,
            'processed': 1,
            'enriched': 1
        },
        {
            'id': 'call_003',
            'title': 'Telnyx Intro Call - Self Labs',
            'company_name': 'Self Labs',
            'date': '2024-02-03',
            'ae_name': 'Bob Johnson',
            'notes': 'AI/ML development company working on SoundGPT voice AI models. Interested in voice capabilities for their AI products. Small team but strong technical background.',
            'action_items_count': 1,
            'follow_up_scheduled': 1,
            'sentiment_score': 7,
            'strategic_score': 7,
            'processed': 1,
            'enriched': 1
        },
        {
            'id': 'call_004',
            'title': 'Telnyx Intro Call - Chuck East',
            'company_name': 'Chuck East',
            'date': '2024-02-04',
            'ae_name': 'Sarah Wilson',
            'notes': 'Small legal practice looking for basic SMS functionality. No immediate budget or timeline. Early stage inquiry.',
            'action_items_count': 0,
            'follow_up_scheduled': 0,
            'sentiment_score': 4,
            'strategic_score': 3,
            'processed': 1,
            'enriched': 1
        },
        {
            'id': 'call_005',
            'title': 'Telnyx Intro Call - Digital MVMT',
            'company_name': 'Digital MVMT',
            'date': '2024-02-05',
            'ae_name': 'Mike Chen',
            'notes': 'CRO agency looking for client notification services. Traditional messaging needs, no AI requirements. Established business but low volume.',
            'action_items_count': 1,
            'follow_up_scheduled': 0,
            'sentiment_score': 6,
            'strategic_score': 5,
            'processed': 1,
            'enriched': 1
        }
    ]
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables (from fellow ingestion script)
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
    
    # Insert sample calls
    for call in sample_calls:
        cursor.execute('''
            INSERT OR REPLACE INTO meetings (
                id, title, company_name, date, ae_name, notes,
                action_items_count, follow_up_scheduled, sentiment_score,
                strategic_score, processed, enriched, raw_data, data_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            call['id'], call['title'], call['company_name'], 
            call['date'], call['ae_name'], call['notes'],
            call['action_items_count'], call['follow_up_scheduled'],
            call['sentiment_score'], call['strategic_score'],
            call['processed'], call['enriched'],
            json.dumps(call), f"hash_{call['id']}"
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created Fellow database with {len(sample_calls)} sample calls")

def train_initial_models():
    """Train initial ML models with sample data"""
    logger.info("Training initial ML models...")
    
    try:
        from feature_engineer import FeatureEngineeringPipeline, load_sample_data
        from model_trainer import train_baseline_model
        
        # Train baseline model
        model_version = train_baseline_model()
        
        logger.info(f"Successfully trained initial models - version: {model_version}")
        return model_version
        
    except Exception as e:
        logger.error(f"Error training initial models: {e}")
        return None

def create_configuration_files():
    """Create configuration files for the system"""
    logger.info("Creating configuration files...")
    
    # API configuration
    api_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "model_version": "baseline",
        "auto_reload": True,
        "max_batch_size": 100,
        "cache_ttl_minutes": 5
    }
    
    with open(PROJECT_ROOT / "config" / "api_config.json", 'w') as f:
        json.dump(api_config, f, indent=2)
    
    # Dashboard configuration
    dashboard_config = {
        "title": "Fellow Learning Qualification Dashboard",
        "refresh_interval_seconds": 300,
        "default_days_back": 30,
        "performance_thresholds": {
            "accuracy_target": 0.85,
            "accuracy_warning": 0.75,
            "precision_target": 0.80,
            "recall_target": 0.70
        }
    }
    
    with open(PROJECT_ROOT / "config" / "dashboard_config.json", 'w') as f:
        json.dump(dashboard_config, f, indent=2)
    
    # Continuous learning configuration
    learning_config = {
        "retrain_frequency_days": 7,
        "drift_check_frequency_days": 1,
        "min_training_samples": 10,
        "performance_thresholds": {
            "accuracy_threshold": 0.75,
            "decline_threshold": 0.1
        },
        "feature_engineering": {
            "max_text_features": 100,
            "min_df": 2,
            "max_df": 0.8
        }
    }
    
    with open(PROJECT_ROOT / "config" / "learning_config.json", 'w') as f:
        json.dump(learning_config, f, indent=2)
    
    logger.info("Created configuration files")

def create_startup_scripts():
    """Create startup scripts for easy system management"""
    logger.info("Creating startup scripts...")
    
    # API startup script
    api_startup = f"""#!/bin/bash
# Start Fellow Learning API Server

cd {PROJECT_ROOT}
export PYTHONPATH=$PYTHONPATH:{PROJECT_ROOT}/ml-model:{PROJECT_ROOT}/api

echo "Starting Fellow Learning API..."
python -m uvicorn api.lead_scorer:app --host 0.0.0.0 --port 8000 --reload
"""
    
    api_script_path = PROJECT_ROOT / "scripts" / "start_api.sh"
    with open(api_script_path, 'w') as f:
        f.write(api_startup)
    api_script_path.chmod(0o755)
    
    # Dashboard startup script
    dashboard_startup = f"""#!/bin/bash
# Start Fellow Learning Dashboard

cd {PROJECT_ROOT}
export PYTHONPATH=$PYTHONPATH:{PROJECT_ROOT}/ml-model:{PROJECT_ROOT}/api

echo "Starting Fellow Learning Dashboard..."
streamlit run dashboard/performance_dashboard.py --server.port 8501
"""
    
    dashboard_script_path = PROJECT_ROOT / "scripts" / "start_dashboard.sh"
    with open(dashboard_script_path, 'w') as f:
        f.write(dashboard_startup)
    dashboard_script_path.chmod(0o755)
    
    # Continuous learning script
    learning_startup = f"""#!/bin/bash
# Run continuous learning cycle

cd {PROJECT_ROOT}
export PYTHONPATH=$PYTHONPATH:{PROJECT_ROOT}/ml-model

echo "Running continuous learning cycle..."
python ml-model/continuous_learner.py
"""
    
    learning_script_path = PROJECT_ROOT / "scripts" / "run_learning.sh"
    with open(learning_script_path, 'w') as f:
        f.write(learning_startup)
    learning_script_path.chmod(0o755)
    
    logger.info("Created startup scripts")

def test_system_components():
    """Test that all system components are working"""
    logger.info("Testing system components...")
    
    test_results = {
        'database': False,
        'feature_engineering': False,
        'model_training': False,
        'api': False,
        'continuous_learning': False
    }
    
    # Test database
    try:
        db_path = PROJECT_ROOT / "data" / "fellow_data.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM meetings")
        count = cursor.fetchone()[0]
        conn.close()
        test_results['database'] = count > 0
        logger.info(f"Database test: {'PASS' if test_results['database'] else 'FAIL'} ({count} records)")
    except Exception as e:
        logger.error(f"Database test failed: {e}")
    
    # Test feature engineering
    try:
        from feature_engineer import FeatureEngineeringPipeline, load_sample_data
        pipeline = FeatureEngineeringPipeline()
        call_data, company_data = load_sample_data()
        features_df = pipeline.prepare_training_data(call_data, company_data)
        test_results['feature_engineering'] = len(features_df) > 0
        logger.info(f"Feature engineering test: {'PASS' if test_results['feature_engineering'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
    
    # Test model files exist
    try:
        models_dir = PROJECT_ROOT / "ml-model" / "models"
        model_versions = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v_')]
        test_results['model_training'] = len(model_versions) > 0
        logger.info(f"Model training test: {'PASS' if test_results['model_training'] else 'FAIL'} ({len(model_versions)} versions)")
    except Exception as e:
        logger.error(f"Model training test failed: {e}")
    
    # Test API import
    try:
        from api.lead_scorer import QualificationAPI, LeadData
        api = QualificationAPI()
        test_results['api'] = api.scorer is not None
        logger.info(f"API test: {'PASS' if test_results['api'] else 'FAIL'}")
    except Exception as e:
        logger.error(f"API test failed: {e}")
    
    # Test continuous learning import
    try:
        from ml_model.continuous_learner import ContinuousLearner
        test_results['continuous_learning'] = True
        logger.info("Continuous learning test: PASS")
    except Exception as e:
        logger.error(f"Continuous learning test failed: {e}")
    
    return test_results

def display_setup_summary(model_version: str = None, test_results: Dict = None):
    """Display setup completion summary"""
    print("\n" + "="*60)
    print("🤖 FELLOW LEARNING QUALIFICATION SYSTEM SETUP COMPLETE!")
    print("="*60)
    
    if model_version:
        print(f"✅ Initial models trained: {model_version}")
    
    if test_results:
        print("\n📋 Component Test Results:")
        for component, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {component.replace('_', ' ').title()}: {status}")
    
    print("\n🚀 Quick Start Commands:")
    print(f"   Start API:       cd {PROJECT_ROOT} && ./scripts/start_api.sh")
    print(f"   Start Dashboard: cd {PROJECT_ROOT} && ./scripts/start_dashboard.sh")
    print(f"   Run Learning:    cd {PROJECT_ROOT} && ./scripts/run_learning.sh")
    
    print("\n🔗 Service URLs:")
    print("   API Documentation: http://localhost:8000/docs")
    print("   Performance Dashboard: http://localhost:8501")
    
    print("\n📁 Key Directories:")
    print(f"   Data: {PROJECT_ROOT}/data/")
    print(f"   Models: {PROJECT_ROOT}/ml-model/models/")
    print(f"   Logs: {PROJECT_ROOT}/logs/")
    print(f"   Config: {PROJECT_ROOT}/config/")
    
    print("\n🎯 Next Steps:")
    print("1. Connect Fellow API for live data ingestion")
    print("2. Integrate with Quinn AI for real-time scoring")
    print("3. Set up automated retraining schedule")
    print("4. Configure monitoring alerts")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 Setting up Fellow Learning Qualification System...")
    
    try:
        # Setup steps
        setup_directory_structure()
        install_requirements()
        create_sample_fellow_database()
        create_configuration_files()
        create_startup_scripts()
        
        # Train initial models
        model_version = train_initial_models()
        
        # Test system
        test_results = test_system_components()
        
        # Display summary
        display_setup_summary(model_version, test_results)
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)