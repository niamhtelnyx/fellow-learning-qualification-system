#!/usr/bin/env python3
"""
Production Qualification Logging System
Comprehensive logging for the entire qualification pipeline

This module provides end-to-end logging for:
- Input capture from Fellow API
- Business Intelligence extraction from calls
- Enrichment process and results  
- ML model scoring and decisions
- Routing logic and assignments
- Outcome tracking and feedback loops
"""

import os
import sys
import json
import uuid
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

# Import business intelligence extractor
from business_intelligence_extractor import extract_business_intelligence_from_fellow, BusinessIntelligence

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Try multiple possible database locations
POSSIBLE_DB_PATHS = [
    os.path.join(DATA_DIR, "fellow_qualification.db"),
    os.path.join(PROJECT_ROOT, "automation", "data", "fellow_qualification.db")
]

# Find existing database or use default path
DB_PATH = POSSIBLE_DB_PATHS[0]  # Default
for path in POSSIBLE_DB_PATHS:
    if os.path.exists(path):
        DB_PATH = path
        break

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Logging setup
logger = logging.getLogger(__name__)

@dataclass
class QualificationMetrics:
    """Performance metrics for qualification pipeline"""
    total_leads: int = 0
    successful_qualifications: int = 0
    failed_qualifications: int = 0
    high_value_leads_found: int = 0
    average_processing_time_ms: float = 0
    enrichment_success_rate: float = 0
    scoring_confidence_avg: float = 0
    routing_success_rate: float = 0

@dataclass
class StageResult:
    """Result of a pipeline stage"""
    stage: str
    status: str  # 'completed', 'failed', 'skipped'
    duration_ms: int
    input_data: Dict
    output_data: Dict
    errors: List[str] = None
    confidence: float = None
    metadata: Dict = None

class QualificationLogger:
    """Main logging class for qualification pipeline"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.current_run_id = None
        self.current_log_id = None
        self.pipeline_version = "2.0.0"
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database exists and is initialized"""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found at {self.db_path}")
            logger.warning("Please run database/setup_database.py first")
            # Create minimal tables for basic operation
            self._create_minimal_tables()
    
    def _create_minimal_tables(self):
        """Create minimal tables if database doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Just the essential tables to get started
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qualification_runs (
                    id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_leads INTEGER DEFAULT 0,
                    successful_qualifications INTEGER DEFAULT 0,
                    failed_qualifications INTEGER DEFAULT 0,
                    pipeline_version TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qualification_logs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    fellow_meeting_id TEXT,
                    company_name TEXT NOT NULL,
                    pipeline_stage TEXT NOT NULL,
                    stage_status TEXT NOT NULL,
                    processing_started TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processing_completed TIMESTAMP,
                    stage_duration_ms INTEGER,
                    input_data TEXT,
                    output_data TEXT,
                    stage_errors TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Created minimal database tables")
            
        except Exception as e:
            logger.error(f"Failed to create minimal tables: {e}")
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def start_qualification_run(self, run_type: str, configuration: Dict = None) -> str:
        """Start a new qualification run and return run_id"""
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO qualification_runs (
                        id, run_type, status, started_at, pipeline_version, configuration
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    run_type,
                    'running',
                    datetime.now().isoformat(),
                    self.pipeline_version,
                    json.dumps(configuration) if configuration else None
                ))
                
                conn.commit()
                
            logger.info(f"Started qualification run: {run_id} (type: {run_type})")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start qualification run: {e}")
            raise
    
    def complete_qualification_run(self, run_id: str = None, metrics: QualificationMetrics = None, error_summary: str = None):
        """Complete a qualification run with final metrics"""
        if not run_id:
            run_id = self.current_run_id
        
        if not run_id:
            logger.warning("No qualification run to complete")
            return
        
        try:
            status = 'completed' if not error_summary else 'failed'
            if metrics and metrics.failed_qualifications > 0 and metrics.successful_qualifications > 0:
                status = 'partial'
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE qualification_runs SET
                        status = ?, completed_at = ?, total_leads = ?,
                        successful_qualifications = ?, failed_qualifications = ?,
                        high_value_leads_found = ?, error_summary = ?,
                        performance_metrics = ?
                    WHERE id = ?
                ''', (
                    status,
                    datetime.now().isoformat(),
                    metrics.total_leads if metrics else 0,
                    metrics.successful_qualifications if metrics else 0,
                    metrics.failed_qualifications if metrics else 0,
                    metrics.high_value_leads_found if metrics else 0,
                    error_summary,
                    json.dumps(asdict(metrics)) if metrics else None,
                    run_id
                ))
                
                conn.commit()
                
            logger.info(f"Completed qualification run: {run_id} (status: {status})")
            
        except Exception as e:
            logger.error(f"Failed to complete qualification run: {e}")
    
    def start_lead_qualification(self, fellow_meeting_id: str, company_name: str, run_id: str = None) -> str:
        """Start qualification logging for a specific lead"""
        if not run_id:
            run_id = self.current_run_id
        
        log_id = str(uuid.uuid4())
        self.current_log_id = log_id
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO qualification_logs (
                        id, run_id, fellow_meeting_id, company_name,
                        pipeline_stage, stage_status, processing_started
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_id,
                    run_id,
                    fellow_meeting_id,
                    company_name,
                    'started',
                    'processing',
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
            logger.info(f"Started lead qualification: {company_name} (log_id: {log_id})")
            return log_id
            
        except Exception as e:
            logger.error(f"Failed to start lead qualification: {e}")
            raise
    
    def log_input_capture(self, log_id: str, raw_fellow_data: Dict, extracted_data: Dict, 
                         data_quality_score: float = None) -> bool:
        """Log input capture stage"""
        return self._log_pipeline_stage(
            log_id=log_id,
            stage='input',
            input_data={'raw_fellow_data': raw_fellow_data},
            output_data=extracted_data,
            confidence=data_quality_score,
            metadata={'extraction_method': extracted_data.get('extraction_method')}
        )
    
    def log_business_intelligence_extraction(self, log_id: str, fellow_data: Dict, transcript: str = None) -> Tuple[bool, Dict]:
        """
        Extract and log business intelligence from Fellow call data
        
        Returns:
            Tuple[success, business_intelligence_dict]
        """
        try:
            start_time = time.time()
            
            # Extract business intelligence
            bi, confidence_scores, overall_confidence = extract_business_intelligence_from_fellow(fellow_data, transcript)
            
            end_time = time.time()
            extraction_time_ms = int((end_time - start_time) * 1000)
            
            # Prepare input data for logging
            input_data = {
                'fellow_data': fellow_data,
                'has_transcript': transcript is not None,
                'transcript_length': len(transcript) if transcript else 0
            }
            
            # Prepare output data
            output_data = bi.to_dict()
            
            # Log the BI extraction stage
            success = self._log_pipeline_stage(
                log_id=log_id,
                stage='business_intelligence',
                input_data=input_data,
                output_data=output_data,
                confidence=overall_confidence,
                metadata={
                    'extraction_method': 'ai_analysis',
                    'extraction_time_ms': extraction_time_ms,
                    'confidence_scores': confidence_scores
                }
            )
            
            # Log detailed BI extraction
            if success:
                self._log_bi_extraction_details(log_id, fellow_data, transcript, bi, confidence_scores, extraction_time_ms)
            
            # Update meetings table with BI data
            self._update_meeting_bi_data(fellow_data.get('fellow_meeting_id'), bi, overall_confidence)
            
            logger.info(f"BI extraction logged for log_id: {log_id} with {overall_confidence:.1f}% confidence")
            return success, output_data
            
        except Exception as e:
            logger.error(f"Failed to extract/log business intelligence: {e}")
            logger.error(traceback.format_exc())
            return False, {}
    
    def log_enrichment_stage(self, log_id: str, enrichment_requests: List[Dict], 
                           enrichment_results: List[Dict], total_cost_cents: int = 0) -> bool:
        """Log enrichment stage with all provider results"""
        
        # Aggregate enrichment data
        input_data = {
            'company_name': enrichment_requests[0].get('company_name') if enrichment_requests else None,
            'enrichment_providers': [req.get('provider') for req in enrichment_requests],
            'total_requests': len(enrichment_requests)
        }
        
        output_data = {
            'enrichment_results': enrichment_results,
            'total_cost_cents': total_cost_cents,
            'successful_enrichments': len([r for r in enrichment_results if r.get('success')]),
            'failed_enrichments': len([r for r in enrichment_results if not r.get('success')])
        }
        
        # Calculate overall confidence
        successful_results = [r for r in enrichment_results if r.get('success')]
        avg_confidence = sum(r.get('confidence_score', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Log individual enrichment attempts
        for request, result in zip(enrichment_requests, enrichment_results):
            self._log_enrichment_attempt(log_id, request, result)
        
        return self._log_pipeline_stage(
            log_id=log_id,
            stage='enrichment',
            input_data=input_data,
            output_data=output_data,
            confidence=avg_confidence,
            metadata={
                'providers_used': list(set(req.get('provider') for req in enrichment_requests)),
                'total_cost_cents': total_cost_cents
            }
        )
    
    def log_scoring_stage(self, log_id: str, model_input: Dict, scoring_result: Dict, 
                         model_performance: Dict = None) -> bool:
        """Log ML model scoring stage"""
        
        return self._log_pipeline_stage(
            log_id=log_id,
            stage='scoring',
            input_data=model_input,
            output_data=scoring_result,
            confidence=scoring_result.get('confidence'),
            metadata={
                'model_name': scoring_result.get('model_name'),
                'model_version': scoring_result.get('model_version'),
                'scoring_method': scoring_result.get('method'),
                'feature_count': len(scoring_result.get('features_used', {})),
                'performance_metrics': model_performance
            }
        )
    
    def log_routing_stage(self, log_id: str, routing_input: Dict, routing_decision: Dict) -> bool:
        """Log routing decision stage"""
        
        return self._log_pipeline_stage(
            log_id=log_id,
            stage='routing',
            input_data=routing_input,
            output_data=routing_decision,
            confidence=routing_decision.get('routing_confidence'),
            metadata={
                'routing_engine': routing_decision.get('routing_engine'),
                'business_rules_applied': routing_decision.get('business_rules_applied'),
                'manual_override': routing_decision.get('manual_override', False)
            }
        )
    
    def log_outcome(self, log_id: str, outcome_type: str, outcome_data: Dict) -> bool:
        """Log qualification outcome"""
        
        return self._log_pipeline_stage(
            log_id=log_id,
            stage='outcome',
            input_data={'outcome_type': outcome_type},
            output_data=outcome_data,
            confidence=outcome_data.get('prediction_accuracy'),
            metadata={
                'data_source': outcome_data.get('data_source'),
                'days_to_outcome': outcome_data.get('days_to_outcome')
            }
        )
    
    def _log_pipeline_stage(self, log_id: str, stage: str, input_data: Dict, 
                           output_data: Dict, confidence: float = None, 
                           metadata: Dict = None, errors: List[str] = None) -> bool:
        """Internal method to log a pipeline stage"""
        try:
            processing_time = time.time() * 1000  # Start time approximation
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update existing log entry or create new stage entry
                cursor.execute('''
                    UPDATE qualification_logs SET
                        pipeline_stage = ?, stage_status = ?, processing_completed = ?,
                        input_data = ?, output_data = ?, confidence_score = ?,
                        metadata = ?, stage_errors = ?
                    WHERE id = ?
                ''', (
                    stage,
                    'completed' if not errors else 'failed',
                    datetime.now().isoformat(),
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    confidence,
                    json.dumps(metadata) if metadata else None,
                    json.dumps(errors) if errors else None,
                    log_id
                ))
                
                # If no rows updated, create new log entry for this stage
                if cursor.rowcount == 0:
                    stage_log_id = str(uuid.uuid4())
                    cursor.execute('''
                        INSERT INTO qualification_logs (
                            id, run_id, fellow_meeting_id, company_name,
                            pipeline_stage, stage_status, processing_started, processing_completed,
                            input_data, output_data, confidence_score, metadata, stage_errors
                        ) SELECT ?, run_id, fellow_meeting_id, company_name, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        FROM qualification_logs WHERE id = ?
                    ''', (
                        stage_log_id,
                        stage,
                        'completed' if not errors else 'failed',
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        json.dumps(input_data) if input_data else None,
                        json.dumps(output_data) if output_data else None,
                        confidence,
                        json.dumps(metadata) if metadata else None,
                        json.dumps(errors) if errors else None,
                        log_id
                    ))
                
                conn.commit()
                
            logger.debug(f"Logged {stage} stage for log_id: {log_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log {stage} stage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _log_enrichment_attempt(self, log_id: str, request: Dict, result: Dict):
        """Log individual enrichment attempt"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if enrichment_logs table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='enrichment_logs'
                """)
                
                if cursor.fetchone():
                    cursor.execute('''
                        INSERT INTO enrichment_logs (
                            id, qualification_log_id, enrichment_provider, enrichment_type,
                            request_data, response_data, success, error_message,
                            response_time_ms, confidence_score, cost_cents, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(uuid.uuid4()),
                        log_id,
                        request.get('provider', 'unknown'),
                        request.get('enrichment_type', 'company_data'),
                        json.dumps(request),
                        json.dumps(result),
                        result.get('success', False),
                        result.get('error_message'),
                        result.get('response_time_ms', 0),
                        result.get('confidence_score', 0),
                        result.get('cost_cents', 0),
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log enrichment attempt: {e}")
    
    def _log_bi_extraction_details(self, log_id: str, fellow_data: Dict, transcript: str, 
                                  bi: BusinessIntelligence, confidence_scores: Dict, extraction_time_ms: int):
        """Log detailed business intelligence extraction"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if business_intelligence_logs table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='business_intelligence_logs'
                """)
                
                if cursor.fetchone():
                    cursor.execute('''
                        INSERT INTO business_intelligence_logs (
                            id, qualification_log_id, extraction_method, transcript_text,
                            raw_extraction, processed_extraction, extraction_model, extraction_prompt,
                            confidence_scores, extraction_time_ms, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(uuid.uuid4()),
                        log_id,
                        'ai_analysis',
                        transcript[:5000] if transcript else None,  # Truncate long transcripts
                        json.dumps(fellow_data),  # Raw extraction input
                        json.dumps(bi.to_dict()),  # Processed BI output
                        'rule_based_v1',  # Current extraction model
                        'business_intelligence_extraction',  # Prompt identifier
                        json.dumps(confidence_scores),
                        extraction_time_ms,
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log BI extraction details: {e}")
    
    def _update_meeting_bi_data(self, meeting_id: str, bi: BusinessIntelligence, confidence: float):
        """Update meetings table with extracted business intelligence"""
        if not meeting_id:
            return
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update meetings table with BI data
                cursor.execute('''
                    UPDATE meetings SET
                        call_context = ?, use_case = ?, products_discussed = ?,
                        ae_next_steps = ?, company_blurb = ?, company_age = ?,
                        employee_count = ?, business_type = ?, business_model = ?,
                        bi_extraction_confidence = ?, bi_extracted_at = ?
                    WHERE id = ?
                ''', (
                    bi.call_context,
                    bi.use_case,
                    json.dumps(bi.products_discussed) if bi.products_discussed else None,
                    bi.ae_next_steps,
                    bi.company_blurb,
                    bi.company_age,
                    bi.employee_count,
                    bi.business_type,
                    bi.business_model,
                    confidence,
                    datetime.now().isoformat(),
                    meeting_id
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update meeting BI data: {e}")
    
    def get_qualification_summary(self, run_id: str = None) -> Dict:
        """Get summary of qualification run"""
        if not run_id:
            run_id = self.current_run_id
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get run summary
                cursor.execute('''
                    SELECT * FROM qualification_runs WHERE id = ?
                ''', (run_id,))
                run_data = cursor.fetchone()
                
                if not run_data:
                    return {}
                
                # Get stage summary
                cursor.execute('''
                    SELECT pipeline_stage, stage_status, COUNT(*) as count,
                           AVG(confidence_score) as avg_confidence
                    FROM qualification_logs 
                    WHERE run_id = ?
                    GROUP BY pipeline_stage, stage_status
                ''', (run_id,))
                
                stage_summary = {}
                for row in cursor.fetchall():
                    stage = row[0]
                    if stage not in stage_summary:
                        stage_summary[stage] = {}
                    stage_summary[stage][row[1]] = {
                        'count': row[2],
                        'avg_confidence': row[3]
                    }
                
                # Get high-value leads
                cursor.execute('''
                    SELECT ql.company_name, ql.confidence_score
                    FROM qualification_logs ql
                    JOIN lead_scores ls ON ql.fellow_meeting_id = ls.meeting_id
                    WHERE ql.run_id = ? AND ls.final_score >= 80
                    ORDER BY ls.final_score DESC
                ''', (run_id,))
                
                high_value_leads = [
                    {'company': row[0], 'confidence': row[1]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    'run_id': run_id,
                    'run_type': run_data[1] if run_data else None,
                    'status': run_data[2] if run_data else None,
                    'started_at': run_data[3] if run_data else None,
                    'completed_at': run_data[4] if run_data else None,
                    'stage_summary': stage_summary,
                    'high_value_leads': high_value_leads
                }
                
        except Exception as e:
            logger.error(f"Failed to get qualification summary: {e}")
            return {}
    
    def get_recent_performance_metrics(self, days_back: int = 7) -> Dict:
        """Get performance metrics for recent qualification runs"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Overall run metrics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_runs,
                        SUM(total_leads) as total_leads,
                        SUM(successful_qualifications) as successful,
                        SUM(failed_qualifications) as failed,
                        SUM(high_value_leads_found) as high_value,
                        AVG(CAST((julianday(completed_at) - julianday(started_at)) * 24 * 60 * 60 * 1000 AS INTEGER)) as avg_runtime_ms
                    FROM qualification_runs
                    WHERE started_at >= ? AND status = 'completed'
                ''', (cutoff_date,))
                
                metrics = cursor.fetchone()
                
                if not metrics or metrics[0] == 0:
                    return {'error': 'No completed runs in specified period'}
                
                # Stage success rates
                cursor.execute('''
                    SELECT 
                        pipeline_stage,
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN stage_status = 'completed' THEN 1 ELSE 0 END) as successful,
                        AVG(confidence_score) as avg_confidence
                    FROM qualification_logs ql
                    JOIN qualification_runs qr ON ql.run_id = qr.id
                    WHERE qr.started_at >= ?
                    GROUP BY pipeline_stage
                ''', (cutoff_date,))
                
                stage_metrics = {}
                for row in cursor.fetchall():
                    stage = row[0]
                    total = row[1]
                    successful = row[2]
                    confidence = row[3]
                    
                    stage_metrics[stage] = {
                        'success_rate': (successful / total * 100) if total > 0 else 0,
                        'total_attempts': total,
                        'avg_confidence': confidence or 0
                    }
                
                return {
                    'period_days': days_back,
                    'total_runs': metrics[0],
                    'total_leads_processed': metrics[1] or 0,
                    'successful_qualifications': metrics[2] or 0,
                    'failed_qualifications': metrics[3] or 0,
                    'high_value_leads_found': metrics[4] or 0,
                    'overall_success_rate': ((metrics[2] or 0) / max(metrics[1] or 1, 1)) * 100,
                    'high_value_rate': ((metrics[4] or 0) / max(metrics[1] or 1, 1)) * 100,
                    'avg_runtime_ms': metrics[5] or 0,
                    'stage_metrics': stage_metrics
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {'error': str(e)}

# Convenience functions for easy integration
def start_qualification_run(run_type: str, configuration: Dict = None) -> Tuple[QualificationLogger, str]:
    """Start a new qualification run and return logger + run_id"""
    logger_instance = QualificationLogger()
    run_id = logger_instance.start_qualification_run(run_type, configuration)
    return logger_instance, run_id

def log_lead_qualification(fellow_meeting_id: str, company_name: str, 
                          qualification_logger: QualificationLogger = None) -> str:
    """Start logging for a specific lead qualification"""
    if not qualification_logger:
        qualification_logger = QualificationLogger()
    
    return qualification_logger.start_lead_qualification(fellow_meeting_id, company_name)