#!/bin/bash
"""
Cron Job Setup Script for Fellow.ai Automation Infrastructure
Sets up automated daily ingestion, enrichment, and monitoring
"""

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
PIPELINES_DIR="$PROJECT_ROOT/pipelines"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Check if running as the correct user
USER=$(whoami)
echo "Setting up Fellow.ai automation cron jobs for user: $USER"

# Backup existing crontab
echo "Backing up existing crontab..."
crontab -l > /tmp/crontab-backup-$(date +%Y%m%d-%H%M%S).txt 2>/dev/null || echo "No existing crontab found"

# Create temporary crontab file
TEMP_CRONTAB="/tmp/fellow-automation-crontab"

# Add existing crontab entries (if any)
crontab -l 2>/dev/null > "$TEMP_CRONTAB" || touch "$TEMP_CRONTAB"

# Add Fellow automation cron jobs
cat >> "$TEMP_CRONTAB" << EOF

# Fellow.ai Automation Infrastructure
# Generated on $(date)

# Daily Fellow API data ingestion (6:00 AM CST)
0 6 * * * cd $PROJECT_ROOT && python3 $SCRIPTS_DIR/fellow-ingestion.py >> $PROJECT_ROOT/logs/cron-ingestion.log 2>&1

# Hourly enrichment pipeline processing (business hours, weekdays)
30 9-17 * * 1-5 cd $PROJECT_ROOT && python3 $PIPELINES_DIR/enrichment-pipeline.py 5 >> $PROJECT_ROOT/logs/cron-enrichment.log 2>&1

# Real-time lead scoring (every 2 hours during business hours)
0 9,11,13,15,17 * * 1-5 cd $PROJECT_ROOT && python3 $SCRIPTS_DIR/realtime-scoring.py >> $PROJECT_ROOT/logs/cron-scoring.log 2>&1

# Health monitoring (every 4 hours)
0 6,10,14,18,22 * * * cd $PROJECT_ROOT && python3 $MONITORING_DIR/health-monitor.py >> $PROJECT_ROOT/logs/cron-health.log 2>&1

# Weekly model retraining (Mondays at 2:00 AM)
0 2 * * 1 cd $PROJECT_ROOT && python3 $SCRIPTS_DIR/model-training.py >> $PROJECT_ROOT/logs/cron-training.log 2>&1

# Daily cleanup and maintenance (3:00 AM)
0 3 * * * cd $PROJECT_ROOT && bash $PROJECT_ROOT/scripts/daily-cleanup.sh >> $PROJECT_ROOT/logs/cron-cleanup.log 2>&1

# Log rotation (weekly, Sundays at 1:00 AM)
0 1 * * 0 find $PROJECT_ROOT/logs -name "*.log" -mtime +30 -delete >> $PROJECT_ROOT/logs/cron-cleanup.log 2>&1

EOF

# Install the new crontab
echo "Installing new crontab..."
crontab "$TEMP_CRONTAB"

# Verify installation
echo "Verifying crontab installation..."
echo "Current crontab entries:"
crontab -l | grep -A 20 "Fellow.ai Automation Infrastructure" || echo "Warning: Fellow automation entries not found in crontab"

# Clean up
rm "$TEMP_CRONTAB"

echo "Cron job setup complete!"
echo ""
echo "Scheduled jobs:"
echo "  - Daily ingestion: 6:00 AM CST"
echo "  - Enrichment: Every 30 minutes past the hour, 9 AM - 5 PM, weekdays"
echo "  - Lead scoring: Every 2 hours, 9 AM - 5 PM, weekdays"
echo "  - Health monitoring: Every 4 hours"
echo "  - Model training: Mondays at 2:00 AM"
echo "  - Cleanup: Daily at 3:00 AM"
echo ""
echo "Log files will be in: $PROJECT_ROOT/logs/"
echo ""
echo "To remove these cron jobs later, run:"
echo "  crontab -e"
echo "  # Remove the Fellow.ai Automation Infrastructure section"