# ML Model Development

This directory contains all machine learning components for the Fellow Learning Qualification System.

## 📁 Directory Structure

### `/data/` - Training Data & Features
- **Raw data**: Fellow call transcripts, company profiles
- **Processed features**: Engineered feature matrices
- **Training sets**: Historical data with outcome labels
- **Validation sets**: Hold-out data for model evaluation

### `/models/` - Model Artifacts
- **Trained models**: Serialized model weights (.joblib, .pkl)
- **Model metadata**: Performance metrics, feature importance
- **Version history**: Model evolution and A/B test results
- **Production models**: Currently deployed model versions

### `/training/` - Training Pipelines
- **Feature engineering**: Data preprocessing and feature extraction
- **Model training**: Training scripts for different algorithms
- **Continuous learning**: Automated retraining pipelines
- **Hyperparameter tuning**: Grid search and optimization

### `/evaluation/` - Performance Monitoring
- **Model evaluation**: Accuracy, precision, recall tracking
- **Performance dashboard**: Real-time monitoring interface
- **Drift detection**: Model degradation alerts
- **A/B testing**: Model comparison frameworks

### `/inference/` - Real-time Scoring
- **Scoring API**: FastAPI service for real-time predictions
- **Batch processing**: High-throughput scoring endpoints
- **Model serving**: Production inference infrastructure
- **Response formatting**: API response standardization

### `/experiments/` - Research & Development
- **Model prototypes**: Experimental architectures
- **Feature research**: New signal discovery
- **Performance analysis**: Deep-dive investigations
- **Ablation studies**: Feature importance analysis

## 🔬 Current Models

### Progression Prediction Model
- **Algorithm**: XGBoost Classifier
- **Purpose**: Predict AE progression likelihood
- **Accuracy**: 85.2% (target: 85%+)
- **Features**: 35+ engineered from company/call data

### Voice AI Detection Model  
- **Algorithm**: Random Forest Classifier
- **Purpose**: Identify Voice AI prospects
- **Precision**: 91.4% (target: 90%+)
- **Priority**: Highest revenue potential prospects

### Qualification Scoring Ensemble
- **Approach**: Weighted combination of models
- **Output**: 0-100 qualification score + routing recommendation
- **Confidence**: Prediction uncertainty quantification

## 📈 Development Workflow

### 1. Data Collection (`/data/`)
```bash
# Fetch new Fellow call data
python training/data_pipeline.py --days-back 30

# Process and engineer features
python training/feature_engineer.py --input raw/ --output processed/
```

### 2. Model Training (`/training/`)
```bash
# Train new model version
python training/model_trainer.py --data processed/features.csv

# Evaluate performance
python evaluation/model_evaluator.py --model-version latest
```

### 3. Model Deployment (`/inference/`)
```bash
# Start scoring API
python inference/lead_scorer.py

# Test scoring endpoint
curl -X POST http://localhost:8000/score -d @test_lead.json
```

### 4. Performance Monitoring (`/evaluation/`)
```bash
# Launch monitoring dashboard
streamlit run evaluation/performance_dashboard.py

# Check for drift
python evaluation/drift_detector.py --model-version production
```

## 🎯 Performance Targets

| Component | Metric | Current | Target |
|-----------|--------|---------|---------|
| **Progression Model** | Accuracy | 85.2% | 85%+ |
| **Voice AI Model** | Precision | 91.4% | 90%+ |
| **Overall System** | Quinn AI Replacement | 38.8% → 85%+ | 2.2x Improvement |
| **API Latency** | Response Time | <200ms | <100ms |

## 🔄 Continuous Learning

The system automatically:
1. **Daily Data Sync**: Fetch new Fellow call outcomes
2. **Feature Engineering**: Process new data into ML-ready format
3. **Model Evaluation**: Check current model performance
4. **Drift Detection**: Alert if accuracy drops below threshold
5. **Automated Retraining**: Update models with new data weekly
6. **A/B Testing**: Compare new vs. current model performance
7. **Production Deployment**: Hot-swap improved models

## 📊 Model Performance Tracking

Performance metrics are tracked in `/evaluation/performance_history.json`:

```json
{
  "model_version": "baseline_v1",
  "training_date": "2024-02-05",
  "metrics": {
    "accuracy": 0.852,
    "precision": 0.831,
    "recall": 0.873,
    "f1_score": 0.851,
    "auc_roc": 0.889
  },
  "feature_importance": [
    {"feature": "voice_ai_signals", "importance": 0.18},
    {"feature": "sentiment_score", "importance": 0.15}
  ]
}
```

## 🚀 Getting Started

```bash
# Setup development environment
pip install -r requirements.txt

# Initialize database and sample data
python training/setup_system.py

# Train initial models
python training/model_trainer.py

# Start API server
python inference/lead_scorer.py

# Launch dashboard
streamlit run evaluation/performance_dashboard.py
```

## 🤝 Contributing

### Commit Standards for ML Development
- **Feature Engineering**: `[feature-eng] Add voice AI detection - 5 new features`
- **Model Training**: `[training] XGBoost model v2 - accuracy: 87.3%`
- **Performance**: `[evaluation] Model drift detection - alert threshold 75%`
- **API Updates**: `[inference] Batch scoring endpoint - 100+ leads/request`

### Code Review Checklist
- [ ] Model performance metrics included in commit
- [ ] Feature importance documented
- [ ] API changes tested with sample data
- [ ] Performance impact assessed
- [ ] Documentation updated