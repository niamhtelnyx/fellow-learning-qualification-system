#!/usr/bin/env python3
"""
Quick Database Query Tool for Fellow Qualification Logs
Easy way to explore the qualification pipeline data
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "data/fellow_qualification.db"

def connect_db():
    return sqlite3.connect(DB_PATH)

def show_qualification_runs():
    """Show all qualification runs"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("🚀 QUALIFICATION RUNS:")
    print("-" * 80)
    cursor.execute("""
        SELECT id, run_type, status, started_at, total_leads, successful_qualifications 
        FROM qualification_runs 
        ORDER BY started_at DESC
    """)
    
    for row in cursor.fetchall():
        print(f"Run ID: {row[0][:8]}... | Type: {row[1]} | Status: {row[2]} | Started: {row[3]} | Leads: {row[4]} | Success: {row[5]}")
    
    conn.close()
    print()

def show_company_pipeline():
    """Show pipeline progression for each company"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("🏢 COMPANY PIPELINE PROGRESSION:")
    print("-" * 80)
    cursor.execute("""
        SELECT DISTINCT company_name 
        FROM qualification_logs 
        ORDER BY company_name
    """)
    
    companies = [row[0] for row in cursor.fetchall()]
    
    for company in companies:
        print(f"\n📊 {company}:")
        cursor.execute("""
            SELECT pipeline_stage, stage_status, stage_duration_ms, confidence_score
            FROM qualification_logs 
            WHERE company_name = ?
            ORDER BY created_at
        """, (company,))
        
        for row in cursor.fetchall():
            stage, status, duration, confidence = row
            duration_str = f"{duration}ms" if duration else "N/A"
            conf_str = f"{confidence:.2f}" if confidence else "N/A"
            print(f"  {stage:12} | {status:10} | {duration_str:8} | confidence: {conf_str}")
    
    conn.close()
    print()

def show_routing_decisions():
    """Show final routing decisions"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("🎯 ROUTING DECISIONS:")
    print("-" * 80)
    cursor.execute("""
        SELECT company_name, routing_type, priority_level, assigned_team, confidence_score
        FROM routing_decisions 
        ORDER BY confidence_score DESC
    """)
    
    for row in cursor.fetchall():
        company, routing, priority, team, confidence = row
        conf_str = f"{confidence:.2f}" if confidence else "N/A"
        print(f"{company:20} | {routing:15} | {priority:12} | {team:15} | confidence: {conf_str}")
    
    conn.close()
    print()

def show_detailed_company(company_name):
    """Show detailed pipeline data for a specific company"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print(f"🔍 DETAILED VIEW: {company_name}")
    print("-" * 80)
    
    # Get qualification logs
    cursor.execute("""
        SELECT pipeline_stage, input_data, output_data, confidence_score
        FROM qualification_logs 
        WHERE company_name = ?
        ORDER BY created_at
    """, (company_name,))
    
    for row in cursor.fetchall():
        stage, input_data, output_data, confidence = row
        print(f"\n📋 Stage: {stage}")
        print(f"   Confidence: {confidence:.2f}" if confidence else "   Confidence: N/A")
        
        if input_data:
            try:
                inp = json.loads(input_data)
                print(f"   Input: {json.dumps(inp, indent=2)[:200]}...")
            except:
                print(f"   Input: {input_data[:100]}...")
        
        if output_data:
            try:
                outp = json.loads(output_data)
                print(f"   Output: {json.dumps(outp, indent=2)[:200]}...")
            except:
                print(f"   Output: {output_data[:100]}...")
    
    conn.close()

def query_db(sql):
    """Run custom SQL query"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("📊 CUSTOM QUERY RESULTS:")
    print("-" * 80)
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # Print column headers
        headers = [description[0] for description in cursor.description]
        print(" | ".join(f"{h:15}" for h in headers))
        print("-" * (16 * len(headers)))
        
        # Print results
        for row in results:
            print(" | ".join(f"{str(cell)[:15]:15}" for cell in row))
            
    except Exception as e:
        print(f"❌ Query error: {e}")
    
    conn.close()
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "runs":
            show_qualification_runs()
        elif command == "companies":
            show_company_pipeline()
        elif command == "routing":
            show_routing_decisions()
        elif command == "detail" and len(sys.argv) > 2:
            show_detailed_company(sys.argv[2])
        elif command == "query" and len(sys.argv) > 2:
            query_db(" ".join(sys.argv[2:]))
        else:
            print("Usage:")
            print("  python query_logs.py runs         # Show qualification runs")
            print("  python query_logs.py companies    # Show company pipeline progression")
            print("  python query_logs.py routing      # Show routing decisions") 
            print("  python query_logs.py detail <company>  # Detailed view of company")
            print("  python query_logs.py query <SQL>  # Custom SQL query")
    else:
        print("📊 FELLOW QUALIFICATION LOGS SUMMARY")
        print("=" * 80)
        show_qualification_runs()
        show_routing_decisions()
        show_company_pipeline()