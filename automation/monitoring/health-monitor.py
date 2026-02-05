#!/usr/bin/env python3
"""
Health Monitoring System for Fellow.ai Automation Infrastructure
Monitors data pipeline health, API connectivity, data quality, and system performance
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
from pathlib import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(DATA_DIR, "fellow_data.db")

# Configuration
FELLOW_API_KEY = "c2e66647b10bfbc93b85cc1b05b8bc519bc61d849a09f5ac8f767fbad927dcc4"
FELLOW_ENDPOINT = "https://telnyx.fellow.app/api/v1/recordings"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
log_file = os.path.join(LOG_DIR, f"health-monitor-{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthMonitorError(Exception):
    """Custom exception for health monitoring errors"""
    pass

class HealthMetrics:
    """Container for health check metrics"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.checks = {}
        self.overall_status = "unknown"
        self.critical_issues = []
        self.warnings = []
        self.info_messages = []
    
    def add_check(self, name: str, status: str, details: dict, critical: bool = False):
        """Add a health check result"""
        self.checks[name] = {
            'status': status,
            'details': details,
            'critical': critical,
            'timestamp': datetime.now().isoformat()
        }
        
        if status == 'error' and critical:
            self.critical_issues.append(f"{name}: {details.get('error', 'Unknown error')}")
        elif status == 'warning':
            self.warnings.append(f"{name}: {details.get('warning', 'Warning condition')}")
        elif status == 'info':
            self.info_messages.append(f"{name}: {details.get('info', 'Information')}")
    
    def calculate_overall_status(self):
        """Calculate overall system health status"""
        if self.critical_issues:
            self.overall_status = "critical"
        elif any(check['status'] == 'error' for check in self.checks.values()):
            self.overall_status = "error"
        elif self.warnings:
            self.overall_status = "warning"
        else:
            self.overall_status = "healthy"
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        self.calculate_overall_status()
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status,
            'checks': self.checks,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'info_messages': self.info_messages,
            'summary': {
                'total_checks': len(self.checks),
                'passed': len([c for c in self.checks.values() if c['status'] == 'ok']),
                'warnings': len(self.warnings),
                'errors': len([c for c in self.checks.values() if c['status'] == 'error']),
                'critical': len(self.critical_issues)
            }
        }

class FellowHealthMonitor:
    """Health monitoring for Fellow.ai automation system"""
    
    def __init__(self):
        self.metrics = HealthMetrics()
        self.session = requests.Session()
        self.session.headers.update({
            'X-Api-Key': FELLOW_API_KEY,
            'Content-Type': 'application/json',
            'User-Agent': 'Telnyx-Fellow-HealthMonitor/1.0'
        })
    
    def check_fellow_api_connectivity(self):
        """Check if Fellow API is accessible"""
        try:
            # Test API with minimal request
            params = {
                'date_range': f"{datetime.now().strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
                'meeting_title': 'Telnyx Intro Call'
            }
            
            start_time = time.time()
            response = self.session.get(FELLOW_ENDPOINT, params=params, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.metrics.add_check('fellow_api_connectivity', 'ok', {
                    'response_time': f"{response_time:.2f}s",
                    'status_code': response.status_code,
                    'recordings_available': len(data.get('recordings', []))
                })
            else:
                self.metrics.add_check('fellow_api_connectivity', 'error', {
                    'status_code': response.status_code,
                    'response_time': f"{response_time:.2f}s",
                    'error': f"HTTP {response.status_code}"
                }, critical=True)
                
        except requests.RequestException as e:
            self.metrics.add_check('fellow_api_connectivity', 'error', {
                'error': str(e),
                'type': 'connection_error'
            }, critical=True)
    
    def check_database_health(self):
        """Check database connectivity and integrity"""
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            cursor = conn.cursor()
            
            # Check if database exists and is accessible
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            required_tables = ['meetings', 'enrichment_queue', 'processing_log']
            missing_tables = [t for t in required_tables if t not in table_names]
            
            if missing_tables:
                self.metrics.add_check('database_health', 'error', {
                    'missing_tables': missing_tables,
                    'error': f"Missing required tables: {', '.join(missing_tables)}"
                }, critical=True)
            else:
                # Check table sizes
                table_stats = {}
                for table in required_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                
                self.metrics.add_check('database_health', 'ok', {
                    'tables': table_stats,
                    'database_size': os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
                })
            
            conn.close()
            
        except sqlite3.Error as e:
            self.metrics.add_check('database_health', 'error', {
                'error': str(e),
                'type': 'database_error'
            }, critical=True)
    
    def check_data_freshness(self):
        """Check if data is being ingested regularly"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check latest data ingestion
            cursor.execute('''
                SELECT MAX(created_at) as latest_meeting, COUNT(*) as total_meetings
                FROM meetings
                WHERE created_at >= date('now', '-7 days')
            ''')
            result = cursor.fetchone()
            latest_meeting = result[0]
            recent_meetings = result[1]
            
            # Check latest processing run
            cursor.execute('''
                SELECT run_type, start_time, status, new_meetings
                FROM processing_log
                ORDER BY start_time DESC
                LIMIT 1
            ''')
            latest_run = cursor.fetchone()
            
            # Evaluate freshness
            if latest_meeting:
                latest_dt = datetime.fromisoformat(latest_meeting)
                hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                
                if hours_old > 48:  # No data in 48 hours
                    self.metrics.add_check('data_freshness', 'error', {
                        'latest_meeting': latest_meeting,
                        'hours_old': f"{hours_old:.1f}",
                        'recent_meetings': recent_meetings,
                        'error': 'Data is stale (>48 hours)'
                    }, critical=True)
                elif hours_old > 24:  # No data in 24 hours
                    self.metrics.add_check('data_freshness', 'warning', {
                        'latest_meeting': latest_meeting,
                        'hours_old': f"{hours_old:.1f}",
                        'recent_meetings': recent_meetings,
                        'warning': 'Data is aging (>24 hours)'
                    })
                else:
                    self.metrics.add_check('data_freshness', 'ok', {
                        'latest_meeting': latest_meeting,
                        'hours_old': f"{hours_old:.1f}",
                        'recent_meetings': recent_meetings
                    })
            else:
                self.metrics.add_check('data_freshness', 'error', {
                    'error': 'No recent meeting data found',
                    'recent_meetings': recent_meetings
                }, critical=True)
            
            # Check processing runs
            if latest_run:
                run_type, start_time, status, new_meetings = latest_run
                run_dt = datetime.fromisoformat(start_time)
                hours_since_run = (datetime.now() - run_dt).total_seconds() / 3600
                
                run_details = {
                    'latest_run_type': run_type,
                    'latest_run_time': start_time,
                    'latest_run_status': status,
                    'hours_since_run': f"{hours_since_run:.1f}",
                    'new_meetings_found': new_meetings
                }
                
                if hours_since_run > 36:  # No processing in 36 hours
                    self.metrics.add_check('processing_runs', 'error', {
                        **run_details,
                        'error': 'No recent processing runs'
                    })
                elif status == 'failure':
                    self.metrics.add_check('processing_runs', 'warning', {
                        **run_details,
                        'warning': 'Latest processing run failed'
                    })
                else:
                    self.metrics.add_check('processing_runs', 'ok', run_details)
            
            conn.close()
            
        except Exception as e:
            self.metrics.add_check('data_freshness', 'error', {
                'error': str(e),
                'type': 'freshness_check_error'
            })
    
    def check_enrichment_pipeline(self):
        """Check enrichment pipeline health"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check enrichment queue status
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM enrichment_queue
                GROUP BY status
            ''')
            queue_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Check error rates
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN error_count > 0 THEN 1 ELSE 0 END) as errors
                FROM enrichment_queue
                WHERE created_at >= date('now', '-7 days')
            ''')
            error_stats = cursor.fetchone()
            total_queue, error_count = error_stats
            
            error_rate = (error_count / total_queue * 100) if total_queue > 0 else 0
            
            # Check recent enrichment activity
            cursor.execute('''
                SELECT COUNT(*) as enriched_today
                FROM enrichment_data
                WHERE created_at >= date('now')
            ''')
            enriched_today = cursor.fetchone()[0]
            
            details = {
                'queue_stats': queue_stats,
                'error_rate': f"{error_rate:.1f}%",
                'enriched_today': enriched_today,
                'total_queued_7d': total_queue
            }
            
            # Evaluate enrichment health
            if error_rate > 50:
                self.metrics.add_check('enrichment_pipeline', 'error', {
                    **details,
                    'error': f'High error rate: {error_rate:.1f}%'
                })
            elif error_rate > 25:
                self.metrics.add_check('enrichment_pipeline', 'warning', {
                    **details,
                    'warning': f'Elevated error rate: {error_rate:.1f}%'
                })
            else:
                self.metrics.add_check('enrichment_pipeline', 'ok', details)
            
            conn.close()
            
        except Exception as e:
            self.metrics.add_check('enrichment_pipeline', 'error', {
                'error': str(e),
                'type': 'enrichment_check_error'
            })
    
    def check_scoring_pipeline(self):
        """Check lead scoring pipeline health"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check for lead_scores table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lead_scores';")
            if not cursor.fetchone():
                self.metrics.add_check('scoring_pipeline', 'warning', {
                    'warning': 'Lead scores table not found - scoring may not be active'
                })
                conn.close()
                return
            
            # Check recent scoring activity
            cursor.execute('''
                SELECT COUNT(*) as scored_today,
                       AVG(final_score) as avg_score,
                       COUNT(CASE WHEN final_score >= 80 THEN 1 END) as high_value_count
                FROM lead_scores
                WHERE created_at >= date('now')
            ''')
            scoring_stats = cursor.fetchone()
            scored_today, avg_score, high_value_count = scoring_stats
            
            # Check scoring coverage
            cursor.execute('''
                SELECT COUNT(*) as total_meetings,
                       COUNT(ls.meeting_id) as scored_meetings
                FROM meetings m
                LEFT JOIN lead_scores ls ON m.id = ls.meeting_id
                WHERE m.created_at >= date('now', '-7 days')
            ''')
            coverage_stats = cursor.fetchone()
            total_meetings, scored_meetings = coverage_stats
            
            coverage_rate = (scored_meetings / total_meetings * 100) if total_meetings > 0 else 0
            
            details = {
                'scored_today': scored_today or 0,
                'avg_score': f"{avg_score:.1f}" if avg_score else "0",
                'high_value_today': high_value_count or 0,
                'coverage_rate': f"{coverage_rate:.1f}%",
                'total_meetings_7d': total_meetings or 0,
                'scored_meetings_7d': scored_meetings or 0
            }
            
            # Evaluate scoring health
            if coverage_rate < 50 and total_meetings > 0:
                self.metrics.add_check('scoring_pipeline', 'warning', {
                    **details,
                    'warning': f'Low scoring coverage: {coverage_rate:.1f}%'
                })
            else:
                self.metrics.add_check('scoring_pipeline', 'ok', details)
            
            conn.close()
            
        except Exception as e:
            self.metrics.add_check('scoring_pipeline', 'error', {
                'error': str(e),
                'type': 'scoring_check_error'
            })
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            # Check data directory disk usage
            data_path = Path(DATA_DIR)
            stat = os.statvfs(data_path)
            
            # Calculate available space
            available_bytes = stat.f_frsize * stat.f_bavail
            total_bytes = stat.f_frsize * stat.f_blocks
            used_bytes = total_bytes - available_bytes
            
            usage_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
            available_gb = available_bytes / (1024 ** 3)
            
            details = {
                'usage_percent': f"{usage_percent:.1f}%",
                'available_gb': f"{available_gb:.2f} GB",
                'data_dir': DATA_DIR
            }
            
            # Evaluate disk space
            if usage_percent > 95 or available_gb < 1:
                self.metrics.add_check('disk_space', 'error', {
                    **details,
                    'error': 'Critical disk space shortage'
                }, critical=True)
            elif usage_percent > 85 or available_gb < 5:
                self.metrics.add_check('disk_space', 'warning', {
                    **details,
                    'warning': 'Low disk space'
                })
            else:
                self.metrics.add_check('disk_space', 'ok', details)
                
        except Exception as e:
            self.metrics.add_check('disk_space', 'error', {
                'error': str(e),
                'type': 'disk_check_error'
            })
    
    def check_log_files(self):
        """Check log file status and errors"""
        try:
            log_files = ['fellow-ingestion', 'enrichment', 'realtime-scoring', 'health-monitor']
            today = datetime.now().strftime('%Y-%m-%d')
            
            log_stats = {}
            recent_errors = []
            
            for log_type in log_files:
                log_file_path = os.path.join(LOG_DIR, f"{log_type}-{today}.log")
                
                if os.path.exists(log_file_path):
                    # Get file size and recent errors
                    file_size = os.path.getsize(log_file_path)
                    
                    # Check for recent errors in log
                    try:
                        with open(log_file_path, 'r') as f:
                            lines = f.readlines()
                            recent_lines = lines[-100:]  # Last 100 lines
                            error_lines = [line for line in recent_lines if 'ERROR' in line]
                            
                            log_stats[log_type] = {
                                'exists': True,
                                'size_bytes': file_size,
                                'recent_errors': len(error_lines)
                            }
                            
                            # Collect recent error messages
                            if error_lines:
                                recent_errors.extend([f"{log_type}: {line.strip()}" for line in error_lines[-3:]])
                    
                    except Exception:
                        log_stats[log_type] = {
                            'exists': True,
                            'size_bytes': file_size,
                            'read_error': True
                        }
                else:
                    log_stats[log_type] = {
                        'exists': False
                    }
            
            details = {
                'log_stats': log_stats,
                'recent_errors': recent_errors[:10]  # Limit to 10 most recent
            }
            
            # Evaluate log health
            missing_logs = [log for log, stats in log_stats.items() if not stats.get('exists')]
            total_recent_errors = sum(stats.get('recent_errors', 0) for stats in log_stats.values())
            
            if missing_logs:
                self.metrics.add_check('log_files', 'warning', {
                    **details,
                    'warning': f'Missing log files: {", ".join(missing_logs)}'
                })
            elif total_recent_errors > 10:
                self.metrics.add_check('log_files', 'warning', {
                    **details,
                    'warning': f'High error count in recent logs: {total_recent_errors}'
                })
            else:
                self.metrics.add_check('log_files', 'ok', details)
                
        except Exception as e:
            self.metrics.add_check('log_files', 'error', {
                'error': str(e),
                'type': 'log_check_error'
            })
    
    def run_health_checks(self) -> dict:
        """Run all health checks and return results"""
        logger.info("Starting comprehensive health check")
        
        # Run all health checks
        self.check_fellow_api_connectivity()
        self.check_database_health()
        self.check_data_freshness()
        self.check_enrichment_pipeline()
        self.check_scoring_pipeline()
        self.check_disk_space()
        self.check_log_files()
        
        # Calculate overall status
        results = self.metrics.to_dict()
        
        logger.info(f"Health check complete - Overall status: {results['overall_status']}")
        if results['critical_issues']:
            logger.error(f"Critical issues found: {len(results['critical_issues'])}")
            for issue in results['critical_issues']:
                logger.error(f"  - {issue}")
        
        return results
    
    def store_health_results(self, results: dict):
        """Store health check results in database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create health_checks table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_status TEXT,
                    results_json TEXT,
                    critical_count INTEGER,
                    warning_count INTEGER,
                    error_count INTEGER
                )
            ''')
            
            # Store results
            cursor.execute('''
                INSERT INTO health_checks (
                    overall_status, results_json, critical_count, warning_count, error_count
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                results['overall_status'],
                json.dumps(results),
                len(results['critical_issues']),
                len(results['warnings']),
                results['summary']['errors']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store health results: {e}")

def main():
    """Main entry point for health monitoring"""
    try:
        monitor = FellowHealthMonitor()
        results = monitor.run_health_checks()
        
        # Store results in database
        monitor.store_health_results(results)
        
        # Output results
        print(json.dumps(results, indent=2))
        
        # Exit with error code if critical issues found
        if results['overall_status'] == 'critical':
            sys.exit(2)
        elif results['overall_status'] == 'error':
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()