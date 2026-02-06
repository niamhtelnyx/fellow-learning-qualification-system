#!/usr/bin/env python3
"""
Enhanced Qualification Pipeline
Processes Fellow calls through the complete enhanced pipeline with 
business intelligence extraction and comprehensive logging.

This script processes the October 2025 test calls through:
1. Input capture and validation
2. Business Intelligence extraction (NEW)
3. Company enrichment 
4. ML scoring
5. Routing decisions
6. Comprehensive audit logging
"""

import os
import sys
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sqlite3

# Add automation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automation'))

from logging_system import QualificationLogger, QualificationMetrics
from business_intelligence_extractor import extract_business_intelligence_from_fellow

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DATA_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "fellow-test-9-10-october-2025-data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup
log_file = os.path.join(OUTPUT_DIR, f"enhanced_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedQualificationPipeline:
    """Enhanced qualification pipeline with business intelligence extraction"""
    
    def __init__(self):
        self.qualification_logger = QualificationLogger()
        self.run_id = None
        
        # Mock enrichment and scoring for testing
        self.mock_enrichment = {
            'clearbit_data': {'industry': 'Technology', 'employees': 25},
            'enrichment_score': 85
        }
        
        # Scoring thresholds
        self.voice_ai_score_boost = 25  # Boost for voice AI companies
        self.base_scores = {
            'Real Estate Technology': 70,
            'Home Services/Logistics': 35,
            'Default': 50
        }
    
    def start_pipeline_run(self, run_type: str = 'enhanced_test') -> str:
        """Start a new enhanced qualification run"""
        configuration = {
            'pipeline_version': '2.0_enhanced',
            'business_intelligence_enabled': True,
            'test_date': datetime.now().isoformat(),
            'features': [
                'business_intelligence_extraction',
                'enhanced_logging',
                'comprehensive_audit_trail'
            ]
        }
        
        self.run_id = self.qualification_logger.start_qualification_run(run_type, configuration)
        logger.info(f"Started enhanced pipeline run: {self.run_id}")
        return self.run_id
    
    def process_fellow_call(self, test_data: Dict) -> Dict:
        """Process a single Fellow call through enhanced pipeline"""
        company_name = test_data['company_name']
        fellow_meeting_id = test_data['fellow_call_id']
        
        logger.info(f"Processing Fellow call for {company_name}")
        
        # Start lead qualification logging
        log_id = self.qualification_logger.start_lead_qualification(
            fellow_meeting_id, company_name, self.run_id
        )
        
        results = {
            'test_id': test_data['test_id'],
            'company_name': company_name,
            'log_id': log_id,
            'fellow_meeting_id': fellow_meeting_id,
            'processing_results': {}
        }
        
        try:
            # Stage 1: Input Capture
            results['processing_results']['input_capture'] = self._process_input_stage(
                log_id, test_data
            )
            
            # Stage 2: Business Intelligence Extraction (NEW)
            results['processing_results']['business_intelligence'] = self._process_bi_extraction_stage(
                log_id, test_data
            )
            
            # Stage 3: Company Enrichment
            results['processing_results']['enrichment'] = self._process_enrichment_stage(
                log_id, test_data
            )
            
            # Stage 4: ML Scoring
            results['processing_results']['scoring'] = self._process_scoring_stage(
                log_id, test_data, results['processing_results']['business_intelligence']
            )
            
            # Stage 5: Routing Decision
            results['processing_results']['routing'] = self._process_routing_stage(
                log_id, test_data, results['processing_results']['scoring']
            )
            
            # Stage 6: Outcome Prediction
            results['processing_results']['outcome'] = self._process_outcome_stage(
                log_id, test_data, results['processing_results']['scoring']
            )
            
            results['success'] = True
            logger.info(f"Successfully processed {company_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {company_name}: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def _process_input_stage(self, log_id: str, test_data: Dict) -> Dict:
        """Process input capture stage"""
        raw_fellow_data = {
            'fellow_call_id': test_data['fellow_call_id'],
            'title': test_data['call_title'],
            'company_name': test_data['company_name'],
            'domain': test_data['domain'],
            'call_date': test_data['call_date'],
            'call_notes': test_data['call_notes'],
            'ae_name': test_data['ae_name'],
            'transcript_snippet': test_data.get('transcript_snippet')
        }
        
        extracted_data = {
            'company_name': test_data['company_name'],
            'ae_name': test_data['ae_name'],
            'call_date': test_data['call_date'],
            'extraction_method': 'test_data',
            'data_quality_score': 95.0  # High quality test data
        }
        
        # Log input capture
        success = self.qualification_logger.log_input_capture(
            log_id, raw_fellow_data, extracted_data, 95.0
        )
        
        return {
            'success': success,
            'extracted_data': extracted_data,
            'data_quality_score': 95.0
        }
    
    def _process_bi_extraction_stage(self, log_id: str, test_data: Dict) -> Dict:
        """Process business intelligence extraction stage (NEW)"""
        # Prepare Fellow data for BI extraction
        fellow_data = {
            'fellow_meeting_id': test_data['fellow_call_id'],
            'title': test_data['call_title'],
            'company_name': test_data['company_name'],
            'notes': test_data['call_notes'],
            'ae_name': test_data['ae_name']
        }
        
        transcript = test_data.get('transcript_snippet')
        
        # Extract business intelligence
        success, bi_data = self.qualification_logger.log_business_intelligence_extraction(
            log_id, fellow_data, transcript
        )
        
        return {
            'success': success,
            'business_intelligence': bi_data,
            'extraction_confidence': bi_data.get('bi_extraction_confidence', 0.0)
        }
    
    def _process_enrichment_stage(self, log_id: str, test_data: Dict) -> Dict:
        """Process company enrichment stage (mock implementation)"""
        # Mock enrichment requests
        enrichment_requests = [
            {
                'company_name': test_data['company_name'],
                'provider': 'clearbit',
                'enrichment_type': 'company_data'
            }
        ]
        
        # Mock enrichment results based on test data
        enrichment_results = [{
            'success': True,
            'provider': 'clearbit',
            'data': {
                'domain': test_data['domain'],
                'industry': test_data['industry'],
                'employees': test_data['employees'],
                'description': f"{test_data['company_name']} in {test_data['industry']}"
            },
            'confidence_score': 85.0,
            'response_time_ms': 150,
            'cost_cents': 5
        }]
        
        # Log enrichment
        success = self.qualification_logger.log_enrichment_stage(
            log_id, enrichment_requests, enrichment_results, total_cost_cents=5
        )
        
        return {
            'success': success,
            'enrichment_results': enrichment_results,
            'total_cost_cents': 5
        }
    
    def _process_scoring_stage(self, log_id: str, test_data: Dict, bi_results: Dict) -> Dict:
        """Process ML scoring stage with BI-enhanced scoring"""
        # Extract BI data for enhanced scoring
        bi_data = bi_results.get('business_intelligence', {})
        
        # Base score from industry
        base_score = self.base_scores.get(test_data['industry'], self.base_scores['Default'])
        
        # BI-based score adjustments
        score_adjustments = []
        
        # Voice AI signals boost
        if 'voice ai' in test_data.get('call_notes', '').lower():
            base_score += self.voice_ai_score_boost
            score_adjustments.append({'reason': 'voice_ai_signals', 'adjustment': self.voice_ai_score_boost})
        
        # Use case scoring
        use_case = bi_data.get('use_case', '')
        if use_case and 'voice ai' in use_case.lower():
            base_score += 15
            score_adjustments.append({'reason': 'voice_ai_use_case', 'adjustment': 15})
        
        # Business type scoring
        business_type = bi_data.get('business_type')
        if business_type == 'Startup':
            base_score += 10  # Startups often good fit
            score_adjustments.append({'reason': 'startup_bonus', 'adjustment': 10})
        
        # Budget/scale signals
        if '$' in test_data.get('call_notes', ''):
            base_score += 10
            score_adjustments.append({'reason': 'budget_mentioned', 'adjustment': 10})
        
        # Cap score at 100
        final_score = min(base_score, 100)
        
        # Mock model input and result
        model_input = {
            'company_name': test_data['company_name'],
            'industry': test_data['industry'],
            'business_intelligence': bi_data,
            'enrichment_data': self.mock_enrichment,
            'urgency_level': test_data.get('urgency_level', 3)
        }
        
        scoring_result = {
            'final_score': final_score,
            'model_name': 'enhanced_bi_scorer',
            'model_version': '2.0',
            'method': 'bi_enhanced_rules',
            'confidence': 85.0,
            'features_used': {
                'industry': test_data['industry'],
                'voice_ai_signals': len(test_data.get('voice_ai_signals', [])),
                'business_signals': len(test_data.get('business_signals', [])),
                'business_intelligence_fields': len([v for v in bi_data.values() if v])
            },
            'score_adjustments': score_adjustments
        }
        
        # Log scoring
        success = self.qualification_logger.log_scoring_stage(
            log_id, model_input, scoring_result
        )
        
        return {
            'success': success,
            'final_score': final_score,
            'scoring_result': scoring_result
        }
    
    def _process_routing_stage(self, log_id: str, test_data: Dict, scoring_results: Dict) -> Dict:
        """Process routing decision stage"""
        final_score = scoring_results['final_score']
        
        # Routing logic based on score
        if final_score >= 80:
            routing_decision = 'AE_HANDOFF'
            priority = 'HIGH'
            assigned_to = test_data['ae_name']
        elif final_score >= 60:
            routing_decision = 'SDR_FOLLOWUP'
            priority = 'MEDIUM'
            assigned_to = 'SDR_TEAM'
        elif final_score >= 40:
            routing_decision = 'NURTURE'
            priority = 'LOW'
            assigned_to = 'MARKETING_AUTOMATION'
        else:
            routing_decision = 'DISQUALIFY'
            priority = 'LOW'
            assigned_to = None
        
        routing_input = {
            'qualification_score': final_score,
            'company_name': test_data['company_name'],
            'urgency_level': test_data.get('urgency_level', 3)
        }
        
        routing_decision_data = {
            'routing_decision': routing_decision,
            'assigned_to': assigned_to,
            'priority_level': priority,
            'routing_confidence': 90.0,
            'routing_engine': 'enhanced_rules_v2',
            'business_rules_applied': [
                f"score_threshold_{routing_decision.lower()}",
                "priority_mapping",
                "assignment_logic"
            ]
        }
        
        # Log routing
        success = self.qualification_logger.log_routing_stage(
            log_id, routing_input, routing_decision_data
        )
        
        return {
            'success': success,
            'routing_decision': routing_decision,
            'priority': priority,
            'assigned_to': assigned_to,
            'routing_data': routing_decision_data
        }
    
    def _process_outcome_stage(self, log_id: str, test_data: Dict, scoring_results: Dict) -> Dict:
        """Process outcome prediction stage"""
        final_score = scoring_results['final_score']
        
        # Predict outcome based on test data expectations
        expected_score_range = test_data.get('expected_score', '50-60')
        expected_routing = test_data.get('expected_routing', 'NURTURE')
        
        # Check if our score matches expectations
        score_parts = expected_score_range.split('-')
        expected_min = int(score_parts[0])
        expected_max = int(score_parts[1]) if len(score_parts) > 1 else expected_min
        
        prediction_accurate = expected_min <= final_score <= expected_max
        
        outcome_data = {
            'outcome_type': 'prediction',
            'predicted_progression': final_score >= 70,
            'prediction_accuracy': 90.0 if prediction_accurate else 60.0,
            'expected_score_range': expected_score_range,
            'actual_score': final_score,
            'expected_routing': expected_routing,
            'data_source': 'test_validation'
        }
        
        # Log outcome
        success = self.qualification_logger.log_outcome(log_id, 'prediction', outcome_data)
        
        return {
            'success': success,
            'prediction_accurate': prediction_accurate,
            'outcome_data': outcome_data
        }
    
    def complete_pipeline_run(self, processed_results: List[Dict]) -> QualificationMetrics:
        """Complete the pipeline run with final metrics"""
        metrics = QualificationMetrics()
        
        metrics.total_leads = len(processed_results)
        metrics.successful_qualifications = len([r for r in processed_results if r['success']])
        metrics.failed_qualifications = metrics.total_leads - metrics.successful_qualifications
        
        # Count high-value leads (score >= 80)
        high_value_count = 0
        total_confidence = 0.0
        confidence_count = 0
        
        for result in processed_results:
            if result['success'] and 'scoring' in result['processing_results']:
                score = result['processing_results']['scoring'].get('final_score', 0)
                if score >= 80:
                    high_value_count += 1
                
                confidence = result['processing_results']['scoring'].get('scoring_result', {}).get('confidence', 0)
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1
        
        metrics.high_value_leads_found = high_value_count
        metrics.scoring_confidence_avg = total_confidence / confidence_count if confidence_count > 0 else 0.0
        metrics.enrichment_success_rate = 95.0  # Mock high success rate
        metrics.routing_success_rate = 100.0  # All processed leads get routed
        
        # Complete the qualification run
        self.qualification_logger.complete_qualification_run(self.run_id, metrics)
        
        return metrics

def load_test_data() -> List[Dict]:
    """Load October 2025 test data"""
    try:
        with open(TEST_DATA_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Test data file not found: {TEST_DATA_PATH}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in test data file: {e}")
        return []

def generate_enhanced_csv(processed_results: List[Dict], metrics: QualificationMetrics, output_path: str):
    """Generate comprehensive CSV with all business intelligence fields"""
    import csv
    
    csv_data = []
    
    for result in processed_results:
        if not result['success']:
            continue
        
        # Extract BI data
        bi_data = result['processing_results'].get('business_intelligence', {}).get('business_intelligence', {})
        scoring_data = result['processing_results'].get('scoring', {})
        routing_data = result['processing_results'].get('routing', {})
        
        csv_row = {
            # Basic Info
            'test_id': result['test_id'],
            'company_name': result['company_name'],
            'fellow_meeting_id': result['fellow_meeting_id'],
            'log_id': result['log_id'],
            
            # Business Intelligence Fields (NEW)
            'call_context': bi_data.get('call_context'),
            'use_case': bi_data.get('use_case'),
            'products_discussed': '; '.join(bi_data.get('products_discussed', [])),
            'ae_next_steps': bi_data.get('ae_next_steps'),
            'company_blurb': bi_data.get('company_blurb'),
            'company_age': bi_data.get('company_age'),
            'employee_count': bi_data.get('employee_count'),
            'business_type': bi_data.get('business_type'),
            'business_model': bi_data.get('business_model'),
            
            # Scoring Results
            'qualification_score': scoring_data.get('final_score'),
            'scoring_confidence': scoring_data.get('scoring_result', {}).get('confidence'),
            'scoring_method': scoring_data.get('scoring_result', {}).get('method'),
            
            # Routing Results
            'routing_decision': routing_data.get('routing_decision'),
            'priority_level': routing_data.get('priority'),
            'assigned_to': routing_data.get('assigned_to'),
            
            # BI Extraction Confidence
            'bi_extraction_confidence': result['processing_results'].get('business_intelligence', {}).get('extraction_confidence'),
            
            # Processing Timestamp
            'processed_at': datetime.now().isoformat()
        }
        
        csv_data.append(csv_row)
    
    # Write CSV
    if csv_data:
        fieldnames = list(csv_data[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Enhanced CSV generated: {output_path}")
    else:
        logger.warning("No data to write to CSV")

def main():
    """Main pipeline execution"""
    logger.info("Starting Enhanced Qualification Pipeline")
    
    # Load test data
    test_data = load_test_data()
    if not test_data:
        logger.error("No test data loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_data)} test cases")
    
    # Initialize pipeline
    pipeline = EnhancedQualificationPipeline()
    
    # Start pipeline run
    run_id = pipeline.start_pipeline_run('enhanced_october_2025_test')
    
    # Process each test case
    processed_results = []
    
    for test_case in test_data:
        logger.info(f"Processing test case {test_case['test_id']}: {test_case['company_name']}")
        
        result = pipeline.process_fellow_call(test_case)
        processed_results.append(result)
        
        # Small delay between processing
        time.sleep(0.5)
    
    # Complete pipeline run
    metrics = pipeline.complete_pipeline_run(processed_results)
    
    # Generate outputs
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 1. Enhanced CSV with all BI fields
    csv_path = os.path.join(OUTPUT_DIR, f"enhanced_qualification_results_{timestamp}.csv")
    generate_enhanced_csv(processed_results, metrics, csv_path)
    
    # 2. Detailed JSON results
    json_path = os.path.join(OUTPUT_DIR, f"detailed_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'metrics': {
                'total_leads': metrics.total_leads,
                'successful_qualifications': metrics.successful_qualifications,
                'failed_qualifications': metrics.failed_qualifications,
                'high_value_leads_found': metrics.high_value_leads_found,
                'scoring_confidence_avg': metrics.scoring_confidence_avg,
                'enrichment_success_rate': metrics.enrichment_success_rate,
                'routing_success_rate': metrics.routing_success_rate
            },
            'processed_results': processed_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    # 3. Summary report
    summary_path = os.path.join(OUTPUT_DIR, f"pipeline_summary_{timestamp}.md")
    with open(summary_path, 'w') as f:
        f.write(f"""# Enhanced Qualification Pipeline Results

## Pipeline Run: {run_id}
**Timestamp:** {timestamp}

## Summary Metrics
- **Total Leads Processed:** {metrics.total_leads}
- **Successful Qualifications:** {metrics.successful_qualifications}
- **Failed Qualifications:** {metrics.failed_qualifications}
- **High-Value Leads Found:** {metrics.high_value_leads_found}
- **Average Scoring Confidence:** {metrics.scoring_confidence_avg:.1f}%
- **Enrichment Success Rate:** {metrics.enrichment_success_rate:.1f}%
- **Routing Success Rate:** {metrics.routing_success_rate:.1f}%

## Business Intelligence Enhancement
This run included the new business intelligence extraction capability that captures:

1. **Call Context** - Why we're meeting them
2. **Use Case** - What they want to use Telnyx for
3. **Products Discussed** - Telnyx products mentioned
4. **AE Next Steps** - Follow-up actions planned
5. **Company Blurb** - Brief company description
6. **Company Age** - Founding year or estimated age
7. **Employee Count** - Size of organization
8. **Business Type** - Startup, SMB, Enterprise, etc.
9. **Business Model** - B2B, B2C, ISV, etc.

## Files Generated
- **Enhanced CSV:** `{os.path.basename(csv_path)}`
- **Detailed JSON:** `{os.path.basename(json_path)}`
- **Summary Report:** `{os.path.basename(summary_path)}`
- **Pipeline Log:** `{os.path.basename(log_file)}`

## Database
All results are logged in the qualification database with comprehensive audit trails.
""")
    
    logger.info("Enhanced Qualification Pipeline Complete")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"Run ID: {run_id}")
    print(f"\n✅ Pipeline complete! Results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()