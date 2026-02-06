-- Production Qualification Logging System Database Schema
-- Version: 2.0
-- Date: 2025-02-05
-- Purpose: Complete end-to-end logging for lead qualification automation

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- =============================================
-- CORE PIPELINE TABLES
-- =============================================

-- 1. QUALIFICATION_RUNS: Track each qualification pipeline execution
CREATE TABLE IF NOT EXISTS qualification_runs (
    id TEXT PRIMARY KEY,                    -- UUID for this qualification run
    run_type TEXT NOT NULL,                 -- 'daily_batch', 'realtime', 'manual', 'backfill'
    status TEXT NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed', 'partial'
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_leads INTEGER DEFAULT 0,
    successful_qualifications INTEGER DEFAULT 0,
    failed_qualifications INTEGER DEFAULT 0,
    high_value_leads_found INTEGER DEFAULT 0,
    pipeline_version TEXT,                  -- Track code version
    configuration TEXT,                     -- JSON: pipeline config used
    error_summary TEXT,                     -- Summary of any errors
    performance_metrics TEXT                -- JSON: timing, memory, etc.
);

-- 2. QUALIFICATION_LOGS: Main table tracking each lead through the pipeline
CREATE TABLE IF NOT EXISTS qualification_logs (
    id TEXT PRIMARY KEY,                    -- UUID for this qualification log entry
    run_id TEXT NOT NULL,                  -- Links to qualification_runs.id
    fellow_meeting_id TEXT,                -- Original Fellow meeting ID
    company_name TEXT NOT NULL,
    pipeline_stage TEXT NOT NULL,          -- 'input', 'enrichment', 'scoring', 'routing', 'outcome'
    stage_status TEXT NOT NULL,            -- 'processing', 'completed', 'failed', 'skipped'
    processing_started TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_completed TIMESTAMP,
    stage_duration_ms INTEGER,             -- Processing time for this stage
    input_data TEXT,                       -- JSON: data entering this stage
    output_data TEXT,                      -- JSON: data exiting this stage
    stage_errors TEXT,                     -- JSON: any errors encountered
    confidence_score REAL,                -- Stage-specific confidence
    metadata TEXT,                         -- JSON: stage-specific metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES qualification_runs (id) ON DELETE CASCADE
);

-- 3. INPUT_CAPTURE: Raw input data from Fellow calls
CREATE TABLE IF NOT EXISTS input_capture (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    fellow_meeting_id TEXT,
    raw_fellow_data TEXT,                  -- JSON: complete Fellow API response
    extracted_company_name TEXT,
    extracted_notes TEXT,
    extracted_ae_name TEXT,
    extracted_date TEXT,
    action_items_count INTEGER,
    follow_up_scheduled BOOLEAN,
    initial_sentiment_score REAL,
    data_quality_score REAL,              -- 0-100: completeness/quality of input
    extraction_method TEXT,               -- How company name was extracted
    extraction_confidence REAL,           -- Confidence in extraction
    data_hash TEXT,                       -- For deduplication
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- 4. ENRICHMENT_LOGS: Track enrichment process and results
CREATE TABLE IF NOT EXISTS enrichment_logs (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    enrichment_provider TEXT NOT NULL,      -- 'clearbit', 'openfunnel', 'web_scraping'
    enrichment_type TEXT NOT NULL,          -- 'company_data', 'ai_signals', 'technology'
    request_data TEXT,                      -- JSON: what we requested
    response_data TEXT,                     -- JSON: what we received
    success BOOLEAN NOT NULL,
    error_message TEXT,
    response_time_ms INTEGER,
    rate_limit_hit BOOLEAN DEFAULT FALSE,
    cache_hit BOOLEAN DEFAULT FALSE,
    data_quality_score REAL,               -- Quality of enriched data
    confidence_score REAL,                 -- Provider's confidence
    cost_cents INTEGER,                    -- API cost tracking
    api_quota_used INTEGER,               -- Quota tracking
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- 5. SCORING_LOGS: Track ML model scoring process
CREATE TABLE IF NOT EXISTS scoring_logs (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    model_name TEXT NOT NULL,              -- 'xgboost_v2', 'random_forest_v1', 'fallback_rules'
    model_version TEXT NOT NULL,
    scoring_method TEXT NOT NULL,          -- 'ml_model', 'fallback_rules', 'emergency_fallback'
    input_features TEXT,                   -- JSON: all features used for scoring
    feature_count INTEGER,
    score INTEGER NOT NULL,                -- 0-100 qualification score
    confidence REAL NOT NULL,              -- Model confidence
    prediction_probability REAL,          -- Raw model probability
    feature_importance TEXT,               -- JSON: feature importance scores
    model_performance_metrics TEXT,        -- JSON: accuracy, precision, etc.
    scoring_time_ms INTEGER,
    memory_usage_mb REAL,
    fallback_reason TEXT,                  -- Why fallback was used
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- 6. ROUTING_DECISIONS: Track routing logic and decisions
CREATE TABLE IF NOT EXISTS routing_decisions (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    routing_engine TEXT NOT NULL,          -- 'rule_based_v1', 'ai_router_v2'
    qualification_score INTEGER NOT NULL,
    voice_ai_fit_score INTEGER,
    progression_probability REAL,
    routing_decision TEXT NOT NULL,        -- 'AE_HANDOFF', 'SDR_FOLLOWUP', 'NURTURE', 'DISQUALIFY'
    assigned_to TEXT,                      -- AE/SDR assigned to
    priority_level TEXT NOT NULL,          -- 'HIGH_VOICE_AI', 'HIGH', 'MEDIUM', 'LOW'
    routing_confidence REAL NOT NULL,
    business_rules_applied TEXT,           -- JSON: which rules triggered
    manual_override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    routing_time_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- 7. OUTCOME_TRACKING: Track actual outcomes vs predictions
CREATE TABLE IF NOT EXISTS outcome_tracking (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    outcome_type TEXT NOT NULL,            -- 'ae_accepted', 'ae_rejected', 'customer_converted', 'lost'
    outcome_date TIMESTAMP,
    actual_progression BOOLEAN,            -- Did lead actually progress?
    deal_value_usd DECIMAL(10,2),         -- Actual deal value if converted
    days_to_outcome INTEGER,              -- Time from qualification to outcome
    ae_feedback TEXT,                     -- AE feedback on lead quality
    ae_rating INTEGER,                    -- AE rating 1-10 on lead quality
    prediction_accuracy REAL,             -- How accurate was our prediction?
    notes TEXT,                           -- Additional outcome notes
    data_source TEXT,                     -- How outcome was captured
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- 8. MODEL_PERFORMANCE: Track model performance over time
CREATE TABLE IF NOT EXISTS model_performance (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    evaluation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    evaluation_period_start TIMESTAMP NOT NULL,
    evaluation_period_end TIMESTAMP NOT NULL,
    total_predictions INTEGER NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_score REAL,
    voice_ai_precision REAL,              -- Specific to voice AI detection
    high_value_precision REAL,            -- Precision for high-value leads
    false_positive_rate REAL,
    false_negative_rate REAL,
    prediction_distribution TEXT,          -- JSON: score distribution
    feature_importance_stability TEXT,     -- JSON: how stable are feature importances
    drift_detection_score REAL,           -- Data drift score
    retrain_recommended BOOLEAN DEFAULT FALSE,
    performance_trend TEXT,               -- 'improving', 'declining', 'stable'
    notes TEXT
);

-- 9. BUSINESS_INTELLIGENCE_LOGS: Track BI extraction process and results
CREATE TABLE IF NOT EXISTS business_intelligence_logs (
    id TEXT PRIMARY KEY,
    qualification_log_id TEXT NOT NULL,
    extraction_method TEXT NOT NULL,        -- 'ai_analysis', 'keyword_extraction', 'rule_based', 'hybrid'
    transcript_text TEXT,                   -- Full or relevant transcript text analyzed
    raw_extraction TEXT,                    -- JSON: raw AI extraction output
    processed_extraction TEXT,              -- JSON: cleaned/processed BI fields
    extraction_model TEXT,                  -- AI model used for extraction
    extraction_prompt TEXT,                 -- Prompt used for AI analysis
    confidence_scores TEXT,                 -- JSON: confidence per BI field
    validation_results TEXT,               -- JSON: validation checks results
    extraction_time_ms INTEGER,            -- Time taken for extraction
    token_usage INTEGER,                   -- AI tokens used
    extraction_cost_cents INTEGER,         -- Cost of extraction
    human_review_required BOOLEAN DEFAULT FALSE, -- Flag for human review
    human_reviewed BOOLEAN DEFAULT FALSE,
    human_review_changes TEXT,             -- JSON: changes made during human review
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (qualification_log_id) REFERENCES qualification_logs (id) ON DELETE CASCADE
);

-- =============================================
-- ENHANCED EXISTING TABLES
-- =============================================

-- Enhanced meetings table (keeping existing + adding business intelligence fields)
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
    enriched BOOLEAN DEFAULT FALSE,
    -- New logging integration fields
    current_qualification_run_id TEXT,      -- Current active qualification run
    last_qualification_log_id TEXT,         -- Last qualification attempt
    qualification_status TEXT,              -- 'pending', 'processing', 'qualified', 'failed'
    total_qualification_attempts INTEGER DEFAULT 0,
    -- NEW BUSINESS INTELLIGENCE FIELDS
    call_context TEXT,                      -- Why are we meeting them (discovery, pricing, technical, follow-up)
    use_case TEXT,                         -- What they want to use Telnyx for
    products_discussed TEXT,               -- JSON array of Telnyx products mentioned/demoed
    ae_next_steps TEXT,                    -- Will AE move forward or not (pricing, tech deep dive, follow-up, self-serve, no fit, rejected traffic)
    company_blurb TEXT,                    -- 1 sentence description of what the company does
    company_age INTEGER,                   -- Year founded or estimated age  
    employee_count TEXT,                   -- Estimated number of employees (range or specific number)
    business_type TEXT,                    -- Startup, SMB, Enterprise, Public, Private, etc.
    business_model TEXT,                   -- B2B, B2C, ISV, etc.
    bi_extraction_confidence REAL,        -- Confidence score for business intelligence extraction (0-100)
    bi_extracted_at TIMESTAMP,            -- When business intelligence was extracted
    
    FOREIGN KEY (current_qualification_run_id) REFERENCES qualification_runs (id),
    FOREIGN KEY (last_qualification_log_id) REFERENCES qualification_logs (id)
);

-- Enhanced enrichment_data table
CREATE TABLE IF NOT EXISTS enrichment_data (
    meeting_id TEXT PRIMARY KEY,
    clearbit_data TEXT,
    openfunnel_data TEXT,
    ai_signals TEXT,
    combined_data TEXT,
    enrichment_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- New logging integration
    last_enrichment_log_id TEXT,            -- Link to enrichment_logs
    enrichment_attempts INTEGER DEFAULT 0,
    enrichment_cost_total_cents INTEGER DEFAULT 0,
    
    FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE,
    FOREIGN KEY (last_enrichment_log_id) REFERENCES qualification_logs (id)
);

-- Enhanced lead_scores table  
CREATE TABLE IF NOT EXISTS lead_scores (
    meeting_id TEXT PRIMARY KEY,
    final_score INTEGER,
    method TEXT,
    model_version TEXT,
    ml_score INTEGER,
    fallback_score INTEGER,
    confidence REAL,
    features_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- New logging integration
    last_scoring_log_id TEXT,               -- Link to scoring_logs
    scoring_attempts INTEGER DEFAULT 0,
    best_score_achieved INTEGER,
    score_stability REAL,                   -- How consistent are scores across attempts
    
    FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE,
    FOREIGN KEY (last_scoring_log_id) REFERENCES qualification_logs (id)
);

-- =============================================
-- SYSTEM TABLES
-- =============================================

-- System configuration and feature flags
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    data_type TEXT NOT NULL,               -- 'string', 'integer', 'boolean', 'json'
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT
);

-- Pipeline health monitoring
CREATE TABLE IF NOT EXISTS pipeline_health (
    id TEXT PRIMARY KEY,
    check_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    component TEXT NOT NULL,               -- 'database', 'fellow_api', 'ml_models', 'enrichment'
    status TEXT NOT NULL,                  -- 'healthy', 'warning', 'critical', 'down'
    response_time_ms INTEGER,
    error_rate REAL,
    throughput_per_hour REAL,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    details TEXT                           -- JSON: component-specific metrics
);

-- =============================================
-- VIEWS FOR EASY QUERYING
-- =============================================

-- Complete qualification pipeline view
CREATE VIEW IF NOT EXISTS v_qualification_pipeline AS
SELECT 
    qr.id as run_id,
    qr.run_type,
    qr.started_at as run_started,
    m.id as meeting_id,
    m.company_name,
    m.ae_name,
    ic.initial_sentiment_score,
    ic.data_quality_score,
    ls.final_score,
    ls.confidence,
    rd.routing_decision,
    rd.priority_level,
    ot.outcome_type,
    ot.actual_progression,
    ot.deal_value_usd,
    CASE 
        WHEN ot.actual_progression = 1 AND ls.final_score >= 80 THEN 'true_positive'
        WHEN ot.actual_progression = 0 AND ls.final_score >= 80 THEN 'false_positive'  
        WHEN ot.actual_progression = 1 AND ls.final_score < 80 THEN 'false_negative'
        WHEN ot.actual_progression = 0 AND ls.final_score < 80 THEN 'true_negative'
        ELSE 'unknown'
    END as prediction_accuracy_category,
    qr.completed_at as run_completed
FROM qualification_runs qr
LEFT JOIN qualification_logs ql ON qr.id = ql.run_id
LEFT JOIN meetings m ON ql.fellow_meeting_id = m.id
LEFT JOIN input_capture ic ON ql.id = ic.qualification_log_id
LEFT JOIN lead_scores ls ON m.id = ls.meeting_id
LEFT JOIN routing_decisions rd ON ql.id = rd.qualification_log_id
LEFT JOIN outcome_tracking ot ON ql.id = ot.qualification_log_id;

-- Model performance summary view
CREATE VIEW IF NOT EXISTS v_model_performance_summary AS
SELECT 
    model_name,
    model_version,
    COUNT(*) as total_predictions,
    AVG(CASE WHEN ot.actual_progression = 1 THEN 1.0 ELSE 0.0 END) as actual_conversion_rate,
    AVG(sl.score) as avg_predicted_score,
    AVG(sl.confidence) as avg_confidence,
    COUNT(CASE WHEN sl.score >= 80 AND ot.actual_progression = 1 THEN 1 END) * 1.0 / 
        NULLIF(COUNT(CASE WHEN sl.score >= 80 THEN 1 END), 0) as precision_high_score,
    COUNT(CASE WHEN ot.actual_progression = 1 AND sl.score >= 80 THEN 1 END) * 1.0 / 
        NULLIF(COUNT(CASE WHEN ot.actual_progression = 1 THEN 1 END), 0) as recall_high_value,
    MAX(sl.created_at) as last_prediction_date
FROM scoring_logs sl
JOIN qualification_logs ql ON sl.qualification_log_id = ql.id
JOIN meetings m ON ql.fellow_meeting_id = m.id
LEFT JOIN outcome_tracking ot ON ql.id = ot.qualification_log_id
WHERE sl.created_at >= datetime('now', '-30 days')
GROUP BY model_name, model_version;

-- =============================================
-- INITIAL CONFIGURATION DATA
-- =============================================

-- System configuration defaults
INSERT OR REPLACE INTO system_config (key, value, data_type, description) VALUES
('pipeline_version', '2.0.0', 'string', 'Current pipeline version'),
('logging_enabled', 'true', 'boolean', 'Enable comprehensive logging'),
('retention_days_logs', '90', 'integer', 'Days to retain detailed logs'),
('retention_days_performance', '365', 'integer', 'Days to retain performance data'),
('enrichment_batch_size', '10', 'integer', 'Batch size for enrichment processing'),
('scoring_confidence_threshold', '0.7', 'string', 'Minimum confidence for ML scores'),
('high_value_score_threshold', '80', 'integer', 'Score threshold for high-value classification'),
('voice_ai_fit_threshold', '85', 'integer', 'Voice AI fit score threshold'),
('max_enrichment_cost_per_lead_cents', '50', 'integer', 'Maximum enrichment cost per lead'),
('model_performance_check_frequency_hours', '24', 'integer', 'Model performance check frequency');

-- =============================================
-- INDEXES FOR PERFORMANCE (Created separately for SQLite compatibility)
-- =============================================

CREATE INDEX IF NOT EXISTS idx_qual_logs_run_id ON qualification_logs (run_id);
CREATE INDEX IF NOT EXISTS idx_qual_logs_company ON qualification_logs (company_name);
CREATE INDEX IF NOT EXISTS idx_qual_logs_stage ON qualification_logs (pipeline_stage);
CREATE INDEX IF NOT EXISTS idx_qual_logs_status ON qualification_logs (stage_status);
CREATE INDEX IF NOT EXISTS idx_qual_logs_time ON qualification_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_qual_logs_composite ON qualification_logs (run_id, pipeline_stage, stage_status);
CREATE INDEX IF NOT EXISTS idx_time_series ON qualification_logs (created_at, pipeline_stage);
CREATE INDEX IF NOT EXISTS idx_company_timeline ON qualification_logs (company_name, created_at);

CREATE INDEX IF NOT EXISTS idx_input_fellow_id ON input_capture (fellow_meeting_id);
CREATE INDEX IF NOT EXISTS idx_input_company ON input_capture (extracted_company_name);
CREATE INDEX IF NOT EXISTS idx_input_hash ON input_capture (data_hash);

CREATE INDEX IF NOT EXISTS idx_enrich_provider ON enrichment_logs (enrichment_provider);
CREATE INDEX IF NOT EXISTS idx_enrich_success ON enrichment_logs (success);
CREATE INDEX IF NOT EXISTS idx_enrich_time ON enrichment_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_enrichment_provider_time ON enrichment_logs (enrichment_provider, created_at);

CREATE INDEX IF NOT EXISTS idx_scoring_model ON scoring_logs (model_name);
CREATE INDEX IF NOT EXISTS idx_scoring_method ON scoring_logs (scoring_method);
CREATE INDEX IF NOT EXISTS idx_scoring_score ON scoring_logs (score);
CREATE INDEX IF NOT EXISTS idx_scoring_time ON scoring_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_scoring_performance ON scoring_logs (model_name, score, confidence);

CREATE INDEX IF NOT EXISTS idx_routing_decision ON routing_decisions (routing_decision);
CREATE INDEX IF NOT EXISTS idx_routing_priority ON routing_decisions (priority_level);
CREATE INDEX IF NOT EXISTS idx_routing_assigned ON routing_decisions (assigned_to);
CREATE INDEX IF NOT EXISTS idx_routing_time ON routing_decisions (created_at);
CREATE INDEX IF NOT EXISTS idx_routing_priority_time ON routing_decisions (priority_level, created_at);

CREATE INDEX IF NOT EXISTS idx_outcome_type ON outcome_tracking (outcome_type);
CREATE INDEX IF NOT EXISTS idx_outcome_progression ON outcome_tracking (actual_progression);
CREATE INDEX IF NOT EXISTS idx_outcome_date ON outcome_tracking (outcome_date);
CREATE INDEX IF NOT EXISTS idx_outcome_value ON outcome_tracking (deal_value_usd);
CREATE INDEX IF NOT EXISTS idx_outcome_conversion ON outcome_tracking (actual_progression, deal_value_usd);

CREATE INDEX IF NOT EXISTS idx_model_perf_name ON model_performance (model_name);
CREATE INDEX IF NOT EXISTS idx_model_perf_date ON model_performance (evaluation_date);
CREATE INDEX IF NOT EXISTS idx_model_perf_accuracy ON model_performance (accuracy);

CREATE INDEX IF NOT EXISTS idx_meetings_qual_status ON meetings (qualification_status);
CREATE INDEX IF NOT EXISTS idx_meetings_attempts ON meetings (total_qualification_attempts);

CREATE INDEX IF NOT EXISTS idx_health_component ON pipeline_health (component);
CREATE INDEX IF NOT EXISTS idx_health_status ON pipeline_health (status);
CREATE INDEX IF NOT EXISTS idx_health_time ON pipeline_health (check_time);

-- =============================================
-- BUSINESS INTELLIGENCE INDEXES
-- =============================================
CREATE INDEX IF NOT EXISTS idx_meetings_call_context ON meetings (call_context);
CREATE INDEX IF NOT EXISTS idx_meetings_use_case ON meetings (use_case);
CREATE INDEX IF NOT EXISTS idx_meetings_ae_next_steps ON meetings (ae_next_steps);
CREATE INDEX IF NOT EXISTS idx_meetings_business_type ON meetings (business_type);
CREATE INDEX IF NOT EXISTS idx_meetings_business_model ON meetings (business_model);
CREATE INDEX IF NOT EXISTS idx_meetings_company_age ON meetings (company_age);
CREATE INDEX IF NOT EXISTS idx_meetings_bi_confidence ON meetings (bi_extraction_confidence);
CREATE INDEX IF NOT EXISTS idx_meetings_bi_extracted ON meetings (bi_extracted_at);

CREATE INDEX IF NOT EXISTS idx_bi_logs_method ON business_intelligence_logs (extraction_method);
CREATE INDEX IF NOT EXISTS idx_bi_logs_model ON business_intelligence_logs (extraction_model);
CREATE INDEX IF NOT EXISTS idx_bi_logs_review ON business_intelligence_logs (human_review_required);
CREATE INDEX IF NOT EXISTS idx_bi_logs_time ON business_intelligence_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_bi_logs_cost ON business_intelligence_logs (extraction_cost_cents);