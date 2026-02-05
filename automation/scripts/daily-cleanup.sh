#!/bin/bash
"""
Daily Cleanup and Maintenance Script
Handles log rotation, data cleanup, and system maintenance
Runs daily at 3:00 AM via cron
"""

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
DB_PATH="$DATA_DIR/fellow_data.db"

# Logging
LOG_FILE="$LOG_DIR/daily-cleanup-$(date +%Y-%m-%d).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "$(date): Starting daily cleanup and maintenance"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1"
}

# Function to check disk space
check_disk_space() {
    log "Checking disk space..."
    
    df -h "$PROJECT_ROOT" | awk 'NR==2 {
        used = $5
        gsub(/%/, "", used)
        if (used > 90) {
            print "WARNING: Disk usage is " used "%"
            exit 1
        } else {
            print "Disk usage: " used "% (OK)"
        }
    }'
}

# Function to rotate logs
rotate_logs() {
    log "Rotating log files..."
    
    # Archive old logs (older than 7 days)
    find "$LOG_DIR" -name "*.log" -mtime +7 -type f | while read -r logfile; do
        if [[ -s "$logfile" ]]; then
            log "Compressing old log: $(basename "$logfile")"
            gzip "$logfile"
        fi
    done
    
    # Remove very old compressed logs (older than 30 days)
    find "$LOG_DIR" -name "*.log.gz" -mtime +30 -type f | while read -r logfile; do
        log "Removing old compressed log: $(basename "$logfile")"
        rm "$logfile"
    done
    
    # Remove empty log files
    find "$LOG_DIR" -name "*.log" -empty -type f -delete
    
    log "Log rotation complete"
}

# Function to cleanup database
cleanup_database() {
    log "Performing database cleanup..."
    
    if [[ ! -f "$DB_PATH" ]]; then
        log "Database not found: $DB_PATH"
        return 1
    fi
    
    # Vacuum database to reclaim space
    sqlite3 "$DB_PATH" "VACUUM;"
    log "Database vacuumed"
    
    # Analyze tables for query optimization
    sqlite3 "$DB_PATH" "ANALYZE;"
    log "Database analyzed"
    
    # Clean up old processing logs (keep last 90 days)
    sqlite3 "$DB_PATH" "DELETE FROM processing_log WHERE start_time < datetime('now', '-90 days');"
    deleted=$(sqlite3 "$DB_PATH" "SELECT changes();")
    log "Cleaned up $deleted old processing log entries"
    
    # Clean up old health check records (keep last 30 days)
    sqlite3 "$DB_PATH" "DELETE FROM health_checks WHERE timestamp < datetime('now', '-30 days');" 2>/dev/null || true
    deleted=$(sqlite3 "$DB_PATH" "SELECT changes();" 2>/dev/null || echo "0")
    log "Cleaned up $deleted old health check records"
    
    # Remove failed enrichment queue items (older than 7 days with 3+ errors)
    sqlite3 "$DB_PATH" "DELETE FROM enrichment_queue WHERE status = 'error' AND error_count >= 3 AND created_at < datetime('now', '-7 days');" 2>/dev/null || true
    deleted=$(sqlite3 "$DB_PATH" "SELECT changes();" 2>/dev/null || echo "0")
    log "Cleaned up $deleted failed enrichment queue items"
    
    log "Database cleanup complete"
}

# Function to backup database
backup_database() {
    log "Creating database backup..."
    
    if [[ ! -f "$DB_PATH" ]]; then
        log "Database not found for backup: $DB_PATH"
        return 1
    fi
    
    BACKUP_DIR="$DATA_DIR/backups"
    mkdir -p "$BACKUP_DIR"
    
    BACKUP_FILE="$BACKUP_DIR/fellow_data_backup_$(date +%Y%m%d).db"
    
    # Create backup only if it doesn't exist today
    if [[ ! -f "$BACKUP_FILE" ]]; then
        cp "$DB_PATH" "$BACKUP_FILE"
        log "Database backed up to: $BACKUP_FILE"
        
        # Compress the backup
        gzip "$BACKUP_FILE"
        log "Backup compressed to: $BACKUP_FILE.gz"
    else
        log "Backup already exists for today: $BACKUP_FILE"
    fi
    
    # Remove old backups (keep last 14 days)
    find "$BACKUP_DIR" -name "fellow_data_backup_*.db.gz" -mtime +14 -delete
    log "Old backups cleaned up (keeping last 14 days)"
}

# Function to check and fix permissions
fix_permissions() {
    log "Checking and fixing file permissions..."
    
    # Ensure proper permissions on directories
    chmod 755 "$PROJECT_ROOT"
    chmod 755 "$DATA_DIR"
    chmod 755 "$LOG_DIR"
    
    # Ensure log files are writable
    find "$LOG_DIR" -name "*.log" -exec chmod 644 {} \;
    
    # Ensure scripts are executable
    find "$PROJECT_ROOT/scripts" -name "*.py" -exec chmod 755 {} \;
    find "$PROJECT_ROOT/scripts" -name "*.sh" -exec chmod 755 {} \;
    find "$PROJECT_ROOT/pipelines" -name "*.py" -exec chmod 755 {} \;
    find "$PROJECT_ROOT/monitoring" -name "*.py" -exec chmod 755 {} \;
    
    log "Permissions fixed"
}

# Function to generate cleanup report
generate_report() {
    log "Generating cleanup report..."
    
    REPORT_FILE="$LOG_DIR/cleanup-report-$(date +%Y-%m-%d).json"
    
    # Get disk usage
    DISK_USAGE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}')
    
    # Get database size
    DB_SIZE=""
    if [[ -f "$DB_PATH" ]]; then
        DB_SIZE=$(stat -f%z "$DB_PATH" 2>/dev/null || stat -c%s "$DB_PATH" 2>/dev/null || echo "unknown")
    fi
    
    # Count log files
    LOG_COUNT=$(find "$LOG_DIR" -name "*.log" -type f | wc -l)
    LOG_SIZE=$(find "$LOG_DIR" -name "*.log" -type f -exec ls -la {} \; | awk '{sum += $5} END {print sum}')
    
    # Get database record counts
    MEETING_COUNT=""
    ENRICHMENT_COUNT=""
    SCORE_COUNT=""
    
    if [[ -f "$DB_PATH" ]]; then
        MEETING_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM meetings;" 2>/dev/null || echo "unknown")
        ENRICHMENT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM enrichment_data;" 2>/dev/null || echo "unknown")
        SCORE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM lead_scores;" 2>/dev/null || echo "unknown")
    fi
    
    # Create JSON report
    cat > "$REPORT_FILE" << EOF
{
    "cleanup_date": "$(date -I)",
    "cleanup_timestamp": "$(date -Iseconds)",
    "disk_usage": "$DISK_USAGE",
    "database": {
        "size_bytes": "$DB_SIZE",
        "meetings": "$MEETING_COUNT",
        "enrichment_records": "$ENRICHMENT_COUNT",
        "lead_scores": "$SCORE_COUNT"
    },
    "logs": {
        "count": $LOG_COUNT,
        "total_size_bytes": "$LOG_SIZE"
    },
    "actions_performed": [
        "log_rotation",
        "database_cleanup",
        "database_backup", 
        "permission_fixes"
    ],
    "status": "completed"
}
EOF
    
    log "Cleanup report generated: $REPORT_FILE"
}

# Function to send alerts if needed
check_alerts() {
    log "Checking for alert conditions..."
    
    # Check disk space
    DISK_USED=$(df "$PROJECT_ROOT" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
    if [[ $DISK_USED -gt 85 ]]; then
        log "ALERT: High disk usage: ${DISK_USED}%"
        # Could integrate with alerting system here
    fi
    
    # Check database size (alert if over 1GB)
    if [[ -f "$DB_PATH" ]]; then
        DB_SIZE_MB=$(stat -f%z "$DB_PATH" 2>/dev/null || stat -c%s "$DB_PATH" 2>/dev/null || echo "0")
        DB_SIZE_MB=$((DB_SIZE_MB / 1024 / 1024))
        
        if [[ $DB_SIZE_MB -gt 1024 ]]; then
            log "ALERT: Large database size: ${DB_SIZE_MB}MB"
        fi
    fi
    
    # Check for recent errors in logs
    ERROR_COUNT=$(find "$LOG_DIR" -name "*$(date +%Y-%m-%d)*" -type f -exec grep -c "ERROR" {} \; 2>/dev/null | awk '{sum += $1} END {print sum+0}')
    if [[ $ERROR_COUNT -gt 10 ]]; then
        log "ALERT: High error count today: $ERROR_COUNT"
    fi
    
    log "Alert check complete"
}

# Main cleanup routine
main() {
    log "=== Daily Cleanup Starting ==="
    
    # Check initial disk space
    check_disk_space || log "WARNING: Disk space check failed"
    
    # Perform cleanup tasks
    rotate_logs || log "WARNING: Log rotation failed"
    backup_database || log "WARNING: Database backup failed"
    cleanup_database || log "WARNING: Database cleanup failed"
    fix_permissions || log "WARNING: Permission fixes failed"
    
    # Generate report and check alerts
    generate_report || log "WARNING: Report generation failed"
    check_alerts || log "WARNING: Alert check failed"
    
    log "=== Daily Cleanup Complete ==="
}

# Run main function
main

# Exit with success
exit 0