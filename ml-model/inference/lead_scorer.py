#!/usr/bin/env python3
"""
Real-time Lead Scoring API for Fellow Learning Qualification System
Provides REST API endpoints for scoring leads in real-time
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import sqlite3
import os
import traceback
from pathlib import Path

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml-model'))

from feature_engineer import FeatureEngineeringPipeline, CompanyFeatureExtractor, CallFeatureExtractor
from model_trainer import QualificationScorer

logger = logging.getLogger(__name__)

# API Models
class LeadData(BaseModel):
    """Input data for lead scoring"""
    # Company information
    company_name: str = Field(..., description="Company name")
    domain: Optional[str] = Field(None, description="Company domain")
    industry: Optional[str] = Field(None, description="Company industry")
    employees: Optional[str] = Field(None, description="Employee count")
    revenue: Optional[str] = Field(None, description="Revenue estimate")
    description: Optional[str] = Field(None, description="Company description")
    
    # Call context
    call_title: Optional[str] = Field(None, description="Meeting title")
    call_notes: Optional[str] = Field(None, description="Call notes/transcript")
    products_discussed: Optional[List[str]] = Field(default=[], description="Products mentioned")
    urgency_level: Optional[int] = Field(0, description="Urgency level (0-5)")
    
    # Additional context
    lead_source: Optional[str] = Field(None, description="Source of lead")
    ae_name: Optional[str] = Field(None, description="AE handling call")

class QualificationResponse(BaseModel):
    """Response model for qualification scoring"""
    lead_id: str
    company_name: str
    qualification_score: int = Field(..., ge=0, le=100)
    voice_ai_fit: int = Field(..., ge=0, le=100)
    progression_probability: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    reasoning: List[str]
    priority: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    scored_at: str

class BatchLeadData(BaseModel):
    """Batch scoring request"""
    leads: List[LeadData]
    batch_id: Optional[str] = None

class BatchQualificationResponse(BaseModel):
    """Batch scoring response"""
    batch_id: str
    results: List[QualificationResponse]
    summary: Dict[str, Any]
    processing_time_ms: float

# API Setup
app = FastAPI(
    title="Fellow Learning Qualification API",
    description="Real-time lead scoring API using ML models trained on Fellow call outcomes",
    version="1.0.0"
)

class QualificationAPI:
    """Main API class for handling scoring requests"""
    
    def __init__(self):
        self.feature_pipeline = None
        self.scorer = None
        self.model_version = None
        self.model_metadata = {}
        self.load_models()
    
    def load_models(self, model_version: str = "baseline"):
        """Load trained models and feature pipeline"""
        try:
            logger.info(f"Loading models version: {model_version}")
            
            # Initialize feature pipeline
            self.feature_pipeline = FeatureEngineeringPipeline()
            
            # Load models
            model_dir = Path(__file__).parent.parent / "ml-model" / "models" / f"v_{model_version}"
            
            if not model_dir.exists():
                # Try to load latest available model
                models_dir = Path(__file__).parent.parent / "ml-model" / "models"
                if models_dir.exists():
                    versions = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v_')]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        model_dir = models_dir / latest_version
                        model_version = latest_version.replace('v_', '')
                        logger.info(f"Using latest available model: {model_version}")
            
            if model_dir.exists():
                # Load metadata
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                
                # Load progression model
                progression_model_path = None
                voice_ai_model_path = None
                
                for model_file in model_dir.glob("*.joblib"):
                    if "progression" in model_file.name:
                        progression_model_path = model_file
                    elif "voice_ai" in model_file.name:
                        voice_ai_model_path = model_file
                
                if progression_model_path and progression_model_path.exists():
                    progression_model = joblib.load(progression_model_path)
                    
                    voice_ai_model = None
                    if voice_ai_model_path and voice_ai_model_path.exists():
                        voice_ai_model = joblib.load(voice_ai_model_path)
                    
                    self.scorer = QualificationScorer(progression_model, voice_ai_model)
                    self.model_version = model_version
                    
                    logger.info(f"Successfully loaded models version {model_version}")
                else:
                    raise FileNotFoundError("Model files not found")
            else:
                logger.warning(f"Model directory not found: {model_dir}")
                self._create_dummy_scorer()
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._create_dummy_scorer()
    
    def _create_dummy_scorer(self):
        """Create a dummy scorer for testing when models aren't available"""
        logger.warning("Creating dummy scorer - models not available")
        self.model_version = "dummy_v1"
        
        class DummyModel:
            def predict_proba(self, X):
                # Random but consistent predictions based on input
                np.random.seed(hash(str(X)) % 2**32)
                return np.array([[0.3, 0.7]])
            
            def predict(self, X):
                prob = self.predict_proba(X)[0, 1]
                return np.array([1 if prob > 0.5 else 0])
        
        dummy_progression = DummyModel()
        dummy_voice_ai = DummyModel()
        
        self.scorer = QualificationScorer(dummy_progression, dummy_voice_ai)
    
    def prepare_features_for_scoring(self, lead_data: LeadData) -> np.ndarray:
        """Convert lead data to feature vector for scoring"""
        try:
            # Create mock call data DataFrame
            call_row = {
                'id': f"api_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'title': lead_data.call_title or f"Call with {lead_data.company_name}",
                'company_name': lead_data.company_name,
                'notes': lead_data.call_notes or "",
                'ae_name': lead_data.ae_name or "API",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'follow_up_scheduled': 0,  # Unknown from API
                'action_items_count': 0,   # Unknown from API
                'sentiment_score': 5,      # Default neutral
                'strategic_score': 5       # Default neutral
            }
            
            call_df = pd.DataFrame([call_row])
            
            # Create mock company data
            company_row = {
                'company': lead_data.company_name,
                'domain': lead_data.domain or "",
                'industry': lead_data.industry or "Unknown",
                'employees': lead_data.employees or "Unknown", 
                'revenue': lead_data.revenue or "Unknown",
                'ai_signals': self._detect_ai_signals(lead_data.description or ""),
                'notes': lead_data.description or ""
            }
            
            company_df = pd.DataFrame([company_row])
            
            # Use feature pipeline to extract features
            features_df = self.feature_pipeline.prepare_training_data(call_df, company_df)
            
            # Fit pipeline if not already fitted (for dummy data)
            if not self.feature_pipeline.is_fitted:
                # Create dummy targets for fitting
                features_df = self.feature_pipeline.create_target_labels(features_df)
                X = self.feature_pipeline.fit_transform(features_df)
            else:
                X = self.feature_pipeline.transform(features_df)
            
            return X[0]  # Return single feature vector
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            logger.error(traceback.format_exc())
            # Return zero feature vector as fallback
            return np.zeros(len(self.feature_pipeline.get_feature_columns()))
    
    def _detect_ai_signals(self, description: str) -> str:
        """Simple AI signal detection from company description"""
        if not description:
            return "Unknown"
        
        desc_lower = description.lower()
        
        voice_ai_keywords = ['voice ai', 'conversational ai', 'ai voice', 'voice automation']
        ai_keywords = ['artificial intelligence', 'machine learning', 'ai', 'automation']
        
        if any(kw in desc_lower for kw in voice_ai_keywords):
            return "Voice AI Primary Business"
        elif any(kw in desc_lower for kw in ai_keywords):
            return "Strong AI Signals"
        else:
            return "Traditional Business"
    
    def score_single_lead(self, lead_data: LeadData, lead_id: str = None) -> QualificationResponse:
        """Score a single lead"""
        if not lead_id:
            lead_id = f"lead_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare features
        features = self.prepare_features_for_scoring(lead_data)
        
        # Score lead
        score_result = self.scorer.score_lead(features, lead_data.company_name)
        
        # Create response
        return QualificationResponse(
            lead_id=lead_id,
            company_name=score_result['company_name'],
            qualification_score=score_result['qualification_score'],
            voice_ai_fit=score_result['voice_ai_fit'],
            progression_probability=score_result['progression_probability'],
            recommendation=score_result['recommendation'],
            reasoning=score_result['reasoning'],
            priority=score_result['priority'],
            confidence=score_result['confidence'],
            model_version=self.model_version,
            scored_at=datetime.now().isoformat()
        )
    
    def score_batch_leads(self, batch_data: BatchLeadData) -> BatchQualificationResponse:
        """Score multiple leads in batch"""
        batch_id = batch_data.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        results = []
        scores = []
        recommendations = {}
        
        for i, lead_data in enumerate(batch_data.leads):
            try:
                lead_id = f"{batch_id}_lead_{i+1}"
                result = self.score_single_lead(lead_data, lead_id)
                results.append(result)
                scores.append(result.qualification_score)
                
                # Count recommendations
                rec = result.recommendation
                recommendations[rec] = recommendations.get(rec, 0) + 1
                
            except Exception as e:
                logger.error(f"Error scoring lead {i+1} in batch {batch_id}: {e}")
                # Add error result
                error_result = QualificationResponse(
                    lead_id=f"{batch_id}_lead_{i+1}_error",
                    company_name=lead_data.company_name,
                    qualification_score=0,
                    voice_ai_fit=0,
                    progression_probability=0.0,
                    recommendation="ERROR",
                    reasoning=[f"Scoring error: {str(e)}"],
                    priority="ERROR",
                    confidence=0.0,
                    model_version=self.model_version,
                    scored_at=datetime.now().isoformat()
                )
                results.append(error_result)
        
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create summary
        summary = {
            "total_leads": len(batch_data.leads),
            "processed_successfully": len([r for r in results if r.recommendation != "ERROR"]),
            "errors": len([r for r in results if r.recommendation == "ERROR"]),
            "average_score": np.mean(scores) if scores else 0,
            "recommendations": recommendations,
            "processing_time_ms": processing_time_ms
        }
        
        return BatchQualificationResponse(
            batch_id=batch_id,
            results=results,
            summary=summary,
            processing_time_ms=processing_time_ms
        )

# Initialize API instance
api_instance = QualificationAPI()

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Fellow Learning Qualification API",
        "version": "1.0.0",
        "model_version": api_instance.model_version,
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": api_instance.scorer is not None,
        "model_version": api_instance.model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/score", response_model=QualificationResponse)
async def score_lead(lead_data: LeadData):
    """Score a single lead"""
    try:
        result = api_instance.score_single_lead(lead_data)
        return result
    except Exception as e:
        logger.error(f"Error scoring lead: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@app.post("/score/batch", response_model=BatchQualificationResponse)
async def score_leads_batch(batch_data: BatchLeadData):
    """Score multiple leads in batch"""
    try:
        if len(batch_data.leads) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 100 leads")
        
        result = api_instance.score_batch_leads(batch_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "model_version": api_instance.model_version,
        "metadata": api_instance.model_metadata,
        "feature_count": len(api_instance.feature_pipeline.get_feature_columns()) if api_instance.feature_pipeline else 0,
        "loaded_at": datetime.now().isoformat()
    }

@app.post("/models/reload/{model_version}")
async def reload_models(model_version: str, background_tasks: BackgroundTasks):
    """Reload models with specified version"""
    try:
        background_tasks.add_task(api_instance.load_models, model_version)
        return {
            "message": f"Model reload initiated for version: {model_version}",
            "status": "reloading"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    # Run the API
    uvicorn.run(
        "lead_scorer:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )