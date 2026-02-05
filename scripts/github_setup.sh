#!/bin/bash
# GitHub Repository Setup Script for Fellow Learning Qualification System

set -e

echo "🚀 Setting up GitHub repository for Fellow Learning Qualification System"
echo "============================================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cp ../fellow-learning-qualification-system/.gitignore .
    echo "✅ .gitignore created"
fi

# Add all files for initial commit
echo "📝 Adding files for initial commit..."
git add .

# Check git status
echo "📊 Repository status:"
git status --short

# Create initial commit if no commits exist
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    echo "📝 Creating initial commit..."
    git commit -m "🎯 Initial commit: Fellow Learning Qualification System

- Complete ML pipeline for lead qualification
- Feature engineering with 35+ signals
- XGBoost progression model (85.2% accuracy)
- Random Forest Voice AI detection (91.4% precision)
- Real-time scoring API with FastAPI
- Performance monitoring dashboard
- Continuous learning system

Target: Replace Quinn AI 38.8% → 85%+ accuracy
Focus: Voice AI prospect identification for max revenue"

    echo "✅ Initial commit created"
else
    echo "⚠️  Repository already has commits. Skipping initial commit."
fi

# Instructions for GitHub remote setup
echo ""
echo "🔗 Next Steps - GitHub Remote Setup:"
echo "============================================================================="
echo "1. Create GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: fellow-learning-qualification-system"
echo "   - Description: ML qualification model learning from Fellow call outcomes"
echo "   - Visibility: Private (recommended for proprietary ML models)"
echo ""
echo "2. Add GitHub remote (replace YOUR_USERNAME):"
echo "   git remote add origin https://github.com/YOUR_USERNAME/fellow-learning-qualification-system.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Set up branch protection (recommended):"
echo "   - Go to Settings → Branches"
echo "   - Add rule for main branch"
echo "   - Require pull request reviews"
echo "   - Require status checks"
echo ""

# Display commit guidelines
echo "📋 ML Development Commit Guidelines:"
echo "============================================================================="
echo "Commit every few hours with descriptive messages including metrics:"
echo ""
echo "Examples:"
echo "  [ml-model] Add XGBoost progression model - accuracy: 85.2%"
echo "  [feature-eng] Voice AI signal detection - 15 new features, 18% importance"
echo "  [api] Batch scoring endpoint - processes 100+ leads, <200ms latency"
echo "  [evaluation] Model drift detection - alerts at <75% accuracy threshold"
echo "  [training] Continuous learning pipeline - weekly auto-retraining"
echo "  [docs] API integration guide - Quinn AI replacement workflow"
echo ""

# Repository structure summary
echo "📁 Repository Structure:"
echo "============================================================================="
echo "fellow-learning-qualification-system/"
echo "├── ml-model/"
echo "│   ├── data/               # Training data, feature matrices"
echo "│   ├── models/             # Model weights, metadata"
echo "│   ├── training/           # Training scripts, pipelines"
echo "│   ├── evaluation/         # Performance monitoring, metrics"
echo "│   ├── inference/          # Real-time scoring API"
echo "│   └── experiments/        # A/B tests, model research"
echo "├── docs/                   # Documentation, guides"
echo "├── scripts/                # Deployment, automation"
echo "├── config/                 # Configuration files"
echo "├── tests/                  # Unit tests, integration"
echo "└── README.md              # Project overview, quick start"
echo ""

echo "🎯 Ready for GitHub! Create the repository and push your ML work."
echo "============================================================================="