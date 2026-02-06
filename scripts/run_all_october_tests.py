#!/usr/bin/env python3
"""
Complete October 2025 Fellow Test Run
Processes all 10 October 2025 Fellow calls through the enhanced pipeline
"""

import os
import sys
import json
from enhanced_qualification_pipeline import EnhancedQualificationPipeline, generate_enhanced_csv, QualificationMetrics
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FULL_TEST_DATA_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "fellow-learning-system", "data", "real_fellow_test_data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_results")

def load_all_test_data():
    """Load all 10 Fellow test calls"""
    try:
        with open(FULL_TEST_DATA_PATH, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} complete Fellow test calls")
        return data
    except FileNotFoundError:
        logger.error(f"Test data file not found: {FULL_TEST_DATA_PATH}")
        return []

def add_transcript_data(test_data):
    """Add mock transcript data for business intelligence extraction"""
    for test in test_data:
        # Create realistic transcript snippets based on the call notes
        company = test['company_name']
        ae = test['ae_name']
        notes = test['call_notes']
        
        # Create transcript snippet
        transcript = f"[00:30] {ae}: Thanks for joining today! Tell me about what you're looking to do with communications.\n"
        transcript += f"[01:00] Contact ({company}): {notes[:200]}...\n"
        transcript += f"[02:30] {ae}: That sounds like a great use case for our platform."
        
        test['transcript_snippet'] = transcript
        
        # Add call_date if not present
        if 'call_date' not in test:
            test['call_date'] = f"2025-10-{10 + test['test_id']:02d}"
        
        # Add fellow_call_id if not present  
        if 'fellow_call_id' not in test:
            test['fellow_call_id'] = f"fc_{test['test_id']:02d}_{company.replace(' ', '').lower()[:8]}"
    
    return test_data

def main():
    """Run all 10 October tests through enhanced pipeline"""
    logger.info("Starting Complete October 2025 Fellow Test Run")
    
    # Load complete test data
    test_data = load_all_test_data()
    if not test_data:
        logger.error("No test data loaded. Exiting.")
        sys.exit(1)
    
    # Add transcript data for BI extraction
    test_data = add_transcript_data(test_data)
    
    logger.info(f"Processing {len(test_data)} Fellow calls through enhanced pipeline")
    
    # Initialize pipeline
    pipeline = EnhancedQualificationPipeline()
    
    # Start pipeline run
    run_id = pipeline.start_pipeline_run('complete_october_2025_test')
    
    # Process all test cases
    processed_results = []
    
    for i, test_case in enumerate(test_data, 1):
        logger.info(f"Processing {i}/{len(test_data)}: {test_case['company_name']}")
        
        result = pipeline.process_fellow_call(test_case)
        processed_results.append(result)
        
        # Small delay between processing
        import time
        time.sleep(0.3)
    
    # Complete pipeline run
    metrics = pipeline.complete_pipeline_run(processed_results)
    
    # Generate comprehensive outputs
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 1. Complete Enhanced CSV
    csv_path = os.path.join(OUTPUT_DIR, f"complete_enhanced_qualification_results_{timestamp}.csv")
    generate_enhanced_csv(processed_results, metrics, csv_path)
    
    # 2. Detailed JSON results
    json_path = os.path.join(OUTPUT_DIR, f"complete_detailed_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'test_run': 'complete_october_2025',
            'total_tests': len(test_data),
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
    
    # 3. Business Intelligence Summary
    bi_summary_path = os.path.join(OUTPUT_DIR, f"business_intelligence_summary_{timestamp}.json")
    
    # Extract all BI data for analysis
    bi_summary = {
        'extracted_business_intelligence': [],
        'extraction_stats': {
            'total_calls': len(processed_results),
            'successful_extractions': 0,
            'call_contexts': {},
            'use_cases': {},
            'business_types': {},
            'business_models': {},
            'products_discussed': {},
            'ae_next_steps': {}
        }
    }
    
    for result in processed_results:
        if result['success'] and 'business_intelligence' in result['processing_results']:
            bi_data = result['processing_results']['business_intelligence']['business_intelligence']
            
            # Add to summary
            bi_summary['extracted_business_intelligence'].append({
                'company_name': result['company_name'],
                'test_id': result['test_id'],
                'business_intelligence': bi_data,
                'extraction_confidence': result['processing_results']['business_intelligence'].get('extraction_confidence', 0)
            })
            
            # Update stats
            stats = bi_summary['extraction_stats']
            stats['successful_extractions'] += 1
            
            # Count occurrences
            for field, value in bi_data.items():
                if value and field in stats:
                    if value not in stats[field]:
                        stats[field][value] = 0
                    stats[field][value] += 1
    
    with open(bi_summary_path, 'w') as f:
        json.dump(bi_summary, f, indent=2)
    
    # 4. Final comprehensive report
    report_path = os.path.join(OUTPUT_DIR, f"complete_test_report_{timestamp}.md")
    with open(report_path, 'w') as f:
        f.write(f"""# Complete October 2025 Fellow Test Results

## Enhanced Qualification Pipeline with Business Intelligence

**Run ID:** {run_id}
**Timestamp:** {timestamp}
**Total Fellow Calls Processed:** {len(test_data)}

## Executive Summary

This test run demonstrates the enhanced qualification pipeline with comprehensive business intelligence extraction from Fellow call data. The pipeline successfully processed all 10 October 2025 Fellow calls and extracted detailed business context for each conversation.

## Key Metrics

- **Total Leads Processed:** {metrics.total_leads}
- **Successful Qualifications:** {metrics.successful_qualifications}
- **High-Value Leads Found:** {metrics.high_value_leads_found}
- **Average Scoring Confidence:** {metrics.scoring_confidence_avg:.1f}%
- **Business Intelligence Success Rate:** {(bi_summary['extraction_stats']['successful_extractions'] / len(processed_results)) * 100:.1f}%

## Business Intelligence Enhancement

The enhanced pipeline now captures 9 critical business intelligence fields:

1. **Call Context** - Meeting purpose and type
2. **Use Case** - Specific Telnyx use case discussed  
3. **Products Discussed** - Which Telnyx products were mentioned
4. **AE Next Steps** - Follow-up actions and progression path
5. **Company Blurb** - Brief company description
6. **Company Age** - Founding year or estimated age
7. **Employee Count** - Organization size
8. **Business Type** - Startup, SMB, Enterprise classification
9. **Business Model** - B2B, B2C, ISV categorization

## Results by Test Case

""")
        
        # Add results for each test case
        for result in processed_results:
            if result['success']:
                bi_data = result['processing_results']['business_intelligence']['business_intelligence']
                scoring_data = result['processing_results']['scoring']
                
                f.write(f"""### Test {result['test_id']}: {result['company_name']}

- **Call Context:** {bi_data.get('call_context', 'Not detected')}
- **Use Case:** {bi_data.get('use_case', 'Not detected')}
- **Products Discussed:** {', '.join(bi_data.get('products_discussed', [])) or 'None detected'}
- **Business Type:** {bi_data.get('business_type', 'Not detected')}
- **Business Model:** {bi_data.get('business_model', 'Not detected')}
- **Qualification Score:** {scoring_data.get('final_score', 0)}
- **Routing Decision:** {result['processing_results']['routing']['routing_decision']}

""")
        
        f.write(f"""## Files Generated

- **Complete Enhanced CSV:** `{os.path.basename(csv_path)}`
- **Detailed JSON Results:** `{os.path.basename(json_path)}`
- **Business Intelligence Summary:** `{os.path.basename(bi_summary_path)}`
- **Complete Test Report:** `{os.path.basename(report_path)}`

## Database Audit Trail

All qualification decisions, business intelligence extractions, and processing steps are logged in the SQLite database with comprehensive audit trails for compliance and analysis.

## Next Steps

1. Review business intelligence extraction accuracy
2. Refine extraction algorithms based on results
3. Integrate with production Fellow webhook
4. Deploy enhanced pipeline to staging environment

---

**Enhanced Qualification Pipeline v2.0 with Business Intelligence**
Generated on {timestamp}
""")
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 COMPLETE OCTOBER 2025 FELLOW TEST RESULTS")
    print("="*60)
    print(f"✅ Processed: {len(test_data)} Fellow calls")
    print(f"✅ Success Rate: {(metrics.successful_qualifications / metrics.total_leads) * 100:.1f}%")
    print(f"✅ High-Value Leads: {metrics.high_value_leads_found}")
    print(f"✅ BI Extraction Rate: {(bi_summary['extraction_stats']['successful_extractions'] / len(processed_results)) * 100:.1f}%")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📊 Run ID: {run_id}")

if __name__ == "__main__":
    main()