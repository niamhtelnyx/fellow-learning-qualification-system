-- Fellow.ai Learning Qualification System Database Schema

-- Create database
CREATE DATABASE IF NOT EXISTS fellow_learning;

-- Fellow calls table
CREATE TABLE fellow_calls (
    id SERIAL PRIMARY KEY,
    fellow_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500),
    date_time TIMESTAMP NOT NULL,
    duration INTEGER, -- in minutes
    participants JSONB,
    transcript TEXT,
    summary TEXT,
    outcome VARCHAR(50),
    ae_progression VARCHAR(20) CHECK (ae_progression IN ('positive', 'neutral', 'negative', 'unknown')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Company profiles table
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    industry VARCHAR(255),
    sub_industry VARCHAR(255),
    employees_count INTEGER,
    revenue_range VARCHAR(50),
    funding_stage VARCHAR(50),
    funding_amount DECIMAL(15,2),
    tech_signals JSONB,
    contact_info JSONB,
    social_presence JSONB,
    competitive_landscape TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Call-Company relationship table
CREATE TABLE call_companies (
    id SERIAL PRIMARY KEY,
    call_id INTEGER REFERENCES fellow_calls(id) ON DELETE CASCADE,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    participant_role VARCHAR(100), -- 'prospect', 'customer', 'partner'
    primary_contact BOOLEAN DEFAULT FALSE,
    UNIQUE(call_id, company_id)
);

-- Enrichment data table
CREATE TABLE enrichments (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    source VARCHAR(50) NOT NULL, -- 'clearbit', 'web_scraping', 'manual'
    data_points JSONB NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    INDEX idx_company_source (company_id, source)
);

-- Call analysis results table
CREATE TABLE call_analyses (
    id SERIAL PRIMARY KEY,
    call_id INTEGER REFERENCES fellow_calls(id) ON DELETE CASCADE,
    context_analysis JSONB, -- meeting purpose, problems, use cases
    products_discussed JSONB, -- Voice AI, Voice, Messaging, etc.
    progression_signals JSONB, -- positive/neutral/negative indicators
    technical_requirements JSONB, -- API needs, volume, compliance
    urgency_signals JSONB, -- timeline and business drivers
    confidence_scores JSONB, -- confidence in each analysis
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyzer_version VARCHAR(20)
);

-- Feature vectors for ML models
CREATE TABLE feature_vectors (
    id SERIAL PRIMARY KEY,
    call_id INTEGER REFERENCES fellow_calls(id) ON DELETE CASCADE,
    feature_vector JSONB NOT NULL,
    feature_names JSONB NOT NULL,
    target_progression VARCHAR(20), -- actual outcome for training
    target_score DECIMAL(5,4), -- numerical target score
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model training history
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'xgboost', 'random_forest', etc.
    hyperparameters JSONB,
    training_data_count INTEGER,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    feature_importance JSONB,
    model_path VARCHAR(500),
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    retired_at TIMESTAMP
);

-- Lead predictions table
CREATE TABLE lead_predictions (
    id SERIAL PRIMARY KEY,
    lead_identifier VARCHAR(255) NOT NULL, -- external lead ID
    company_domain VARCHAR(255),
    prediction_score DECIMAL(5,4) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    reasoning JSONB, -- explanation of score factors
    predicted_progression VARCHAR(20),
    model_version VARCHAR(20) REFERENCES model_versions(version),
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_outcome VARCHAR(20), -- filled when outcome known
    outcome_time TIMESTAMP,
    INDEX idx_lead_domain (company_domain),
    INDEX idx_prediction_time (prediction_time)
);

-- System performance metrics
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_data JSONB,
    model_version VARCHAR(20),
    time_period_start TIMESTAMP,
    time_period_end TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_time (metric_name, recorded_at)
);

-- Data processing jobs table
CREATE TABLE processing_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL, -- 'fetch_calls', 'enrich_company', 'train_model'
    job_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    job_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_fellow_calls_date ON fellow_calls(date_time);
CREATE INDEX idx_fellow_calls_outcome ON fellow_calls(ae_progression);
CREATE INDEX idx_companies_domain ON companies(domain);
CREATE INDEX idx_companies_industry ON companies(industry);
CREATE INDEX idx_call_analyses_call_id ON call_analyses(call_id);
CREATE INDEX idx_feature_vectors_call_id ON feature_vectors(call_id);
CREATE INDEX idx_enrichments_company_id ON enrichments(company_id);
CREATE INDEX idx_enrichments_expires ON enrichments(expires_at);

-- Create updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_fellow_calls_updated_at 
    BEFORE UPDATE ON fellow_calls 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_companies_updated_at 
    BEFORE UPDATE ON companies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW enriched_companies AS
SELECT 
    c.*,
    e.data_points as latest_enrichment,
    e.confidence_score as enrichment_confidence,
    e.enriched_at as last_enriched
FROM companies c
LEFT JOIN LATERAL (
    SELECT * FROM enrichments 
    WHERE company_id = c.id 
    ORDER BY enriched_at DESC 
    LIMIT 1
) e ON true;

CREATE VIEW analyzed_calls AS
SELECT 
    fc.*,
    ca.context_analysis,
    ca.products_discussed,
    ca.progression_signals,
    ca.technical_requirements,
    ca.urgency_signals,
    ca.confidence_scores,
    ca.analyzed_at
FROM fellow_calls fc
LEFT JOIN call_analyses ca ON fc.id = ca.call_id;

CREATE VIEW model_performance_summary AS
SELECT 
    mv.version,
    mv.model_type,
    mv.accuracy,
    mv.precision_score,
    mv.recall_score,
    mv.f1_score,
    mv.trained_at,
    COUNT(lp.id) as predictions_made,
    AVG(lp.confidence_score) as avg_confidence
FROM model_versions mv
LEFT JOIN lead_predictions lp ON mv.version = lp.model_version
GROUP BY mv.id, mv.version, mv.model_type, mv.accuracy, mv.precision_score, 
         mv.recall_score, mv.f1_score, mv.trained_at;

-- Insert initial model version
INSERT INTO model_versions (
    version, 
    model_type, 
    hyperparameters, 
    training_data_count, 
    accuracy, 
    model_path
) VALUES (
    'baseline_v1.0', 
    'logistic_regression', 
    '{"C": 1.0, "random_state": 42}', 
    0, 
    0.0, 
    'models/baseline_v1.0.pkl'
);