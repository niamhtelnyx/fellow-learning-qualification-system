#!/usr/bin/env python3
"""
INTEGRATE ENHANCED "SHOW ME YOU KNOW ME" DESCRIPTIONS V2
Update the qualification_logs with enhanced company blurbs and use case integrations
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
import os

class EnhancedDescriptionIntegratorV2:
    def __init__(self, db_path="data/fellow_qualification.db"):
        self.db_path = db_path
        self.enhanced_data = None
        
    def backup_database(self):
        """Create backup of current database"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = f"data/fellow_qualification_backup_v2_{timestamp}.db"
        
        # Copy database file
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
        
    def check_qualification_logs_schema(self):
        """Check current schema of qualification_logs table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(qualification_logs)")
        columns = cursor.fetchall()
        
        print("📋 Current qualification_logs schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            
        conn.close()
        return [col[1] for col in columns]
        
    def update_qualification_logs_schema(self):
        """Add new columns for enhanced descriptions to qualification_logs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add enhanced description columns
        schema_updates = [
            "ALTER TABLE qualification_logs ADD COLUMN enhanced_company_blurb TEXT",
            "ALTER TABLE qualification_logs ADD COLUMN enhanced_use_case_integration TEXT", 
            "ALTER TABLE qualification_logs ADD COLUMN enhancement_method TEXT DEFAULT 'show_me_you_know_me_v2'",
            "ALTER TABLE qualification_logs ADD COLUMN enhancement_confidence REAL",
            "ALTER TABLE qualification_logs ADD COLUMN enhanced_at TIMESTAMP"
        ]
        
        for sql in schema_updates:
            try:
                cursor.execute(sql)
                column_name = sql.split('ADD COLUMN')[1].split()[0] if 'ADD COLUMN' in sql else 'N/A'
                print(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e):
                    column_name = sql.split('ADD COLUMN')[1].split()[0] if 'ADD COLUMN' in sql else 'N/A'
                    print(f"⚠️ Column already exists: {column_name}")
                else:
                    print(f"❌ Schema update failed: {e}")
                    
        conn.commit()
        conn.close()
        
    def load_enhanced_data(self, csv_path):
        """Load enhanced descriptions from CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Enhanced data file not found: {csv_path}")
            
        self.enhanced_data = pd.read_csv(csv_path)
        print(f"✅ Loaded enhanced data: {len(self.enhanced_data)} companies")
        return self.enhanced_data
        
    def update_qualification_records(self):
        """Update qualification_logs records with enhanced descriptions"""
        if self.enhanced_data is None:
            raise ValueError("No enhanced data loaded. Call load_enhanced_data() first.")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updated_count = 0
        
        for _, row in self.enhanced_data.iterrows():
            fellow_meeting_id = row['fellow_meeting_id']
            
            # Update all qualification_logs records for this meeting
            update_sql = """
            UPDATE qualification_logs 
            SET enhanced_company_blurb = ?,
                enhanced_use_case_integration = ?,
                enhancement_method = 'show_me_you_know_me_v2',
                enhancement_confidence = ?,
                enhanced_at = ?
            WHERE fellow_meeting_id = ?
            """
            
            cursor.execute(update_sql, (
                row['enhanced_company_blurb'],
                row['enhanced_use_case_integration'], 
                row['enhancement_confidence'],
                datetime.now().isoformat(),
                fellow_meeting_id
            ))
            
            if cursor.rowcount > 0:
                updated_count += cursor.rowcount
                print(f"✅ Updated {cursor.rowcount} records for: {row['company_name']} ({fellow_meeting_id})")
            else:
                print(f"⚠️ No records found for: {fellow_meeting_id}")
                
        conn.commit()
        conn.close()
        
        print(f"🎯 Total qualification records updated: {updated_count}")
        return updated_count
        
    def generate_enhanced_qualification_export(self, output_path="test_results/enhanced_qualification_with_smykm_descriptions.csv"):
        """Generate comprehensive export with enhanced descriptions"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT DISTINCT
            fellow_meeting_id,
            company_name,
            MAX(company_blurb) as original_company_blurb,
            MAX(enhanced_company_blurb) as enhanced_company_blurb,
            MAX(use_case) as original_use_case,
            MAX(enhanced_use_case_integration) as enhanced_use_case_integration,
            MAX(products_discussed) as products_discussed,
            MAX(ae_next_steps) as ae_next_steps,
            MAX(call_context) as call_context,
            MAX(business_type) as business_type,
            MAX(business_model) as business_model,
            MAX(employee_count) as employee_count,
            MAX(company_age) as company_age,
            MAX(qualification_score) as qualification_score,
            MAX(expected_routing) as expected_routing,
            MAX(enhancement_confidence) as enhancement_confidence,
            MAX(enhanced_at) as enhanced_at,
            MAX(processed_at) as processed_at
        FROM qualification_logs 
        WHERE enhanced_company_blurb IS NOT NULL
        GROUP BY fellow_meeting_id, company_name
        ORDER BY enhanced_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Calculate improvement metrics
        df['blurb_length_improvement'] = df['enhanced_company_blurb'].str.len() - df['original_company_blurb'].fillna('').str.len()
        df['use_case_length_improvement'] = df['enhanced_use_case_integration'].str.len() - df['original_use_case'].fillna('').str.len()
        
        df.to_csv(output_path, index=False)
        print(f"✅ Enhanced qualification export generated: {output_path}")
        print(f"📊 Records exported: {len(df)}")
        
        return df
        
    def create_comparison_display(self, df):
        """Create side-by-side comparison display"""
        print("\n🎯 ENHANCED 'SHOW ME YOU KNOW ME' COMPARISON")
        print("=" * 70)
        
        for _, row in df.iterrows():
            print(f"\n🏢 {row['company_name']}")
            print(f"📅 Fellow Meeting ID: {row['fellow_meeting_id']}")
            print(f"⭐ Qualification Score: {row['qualification_score']}")
            
            print(f"\n📝 ORIGINAL COMPANY DESCRIPTION:")
            print(f"   {row['original_company_blurb'] or 'No description'}")
            
            print(f"\n✨ ENHANCED 'SHOW ME YOU KNOW ME' DESCRIPTION:")
            print(f"   {row['enhanced_company_blurb']}")
            
            print(f"\n📦 ORIGINAL USE CASE:")
            print(f"   {row['original_use_case'] or 'No use case defined'}")
            
            print(f"\n🔧 ENHANCED TELNYX INTEGRATION WORKFLOW:")
            print(f"   {row['enhanced_use_case_integration']}")
            
            print(f"\n📈 METRICS:")
            print(f"   • Blurb improvement: +{row['blurb_length_improvement']:.0f} characters")
            print(f"   • Use case improvement: +{row['use_case_length_improvement']:.0f} characters")
            print(f"   • Enhancement confidence: {row['enhancement_confidence']:.1f}%")
            
            print("=" * 70)

def main():
    """Main integration workflow"""
    print("🔧 ENHANCED 'SHOW ME YOU KNOW ME' DATABASE INTEGRATION V2")
    print("=" * 65)
    
    # Initialize integrator
    integrator = EnhancedDescriptionIntegratorV2()
    
    # Step 1: Backup database
    print("\n📦 Step 1: Creating database backup...")
    backup_path = integrator.backup_database()
    
    # Step 2: Check current schema
    print("\n🔍 Step 2: Checking current schema...")
    current_columns = integrator.check_qualification_logs_schema()
    
    # Step 3: Update schema
    print("\n🏗️ Step 3: Updating qualification_logs schema...")
    integrator.update_qualification_logs_schema()
    
    # Step 4: Load enhanced data
    print("\n📊 Step 4: Loading enhanced descriptions...")
    # Find the most recent enhanced CSV
    import glob
    enhanced_files = glob.glob("test_results/show_me_you_know_me_enhanced_*.csv")
    if not enhanced_files:
        print("❌ No enhanced data files found!")
        return
        
    latest_file = max(enhanced_files, key=os.path.getctime)
    print(f"📁 Using latest file: {latest_file}")
    
    integrator.load_enhanced_data(latest_file)
    
    # Step 5: Update qualification records
    print("\n🔄 Step 5: Updating qualification records...")
    updated_count = integrator.update_qualification_records()
    
    # Step 6: Generate enhanced export
    print("\n📤 Step 6: Generating enhanced qualification export...")
    enhanced_df = integrator.generate_enhanced_qualification_export()
    
    # Step 7: Display comparison
    print("\n📋 Step 7: Displaying enhanced descriptions...")
    integrator.create_comparison_display(enhanced_df)
    
    # Summary statistics
    print("\n📊 INTEGRATION SUMMARY")
    print("=" * 35)
    print(f"✅ Qualification records enhanced: {updated_count}")
    print(f"🏢 Companies processed: {len(enhanced_df)}")
    print(f"📈 Average blurb length increase: {enhanced_df['blurb_length_improvement'].mean():.0f} characters")
    print(f"🎯 Average confidence score: {enhanced_df['enhancement_confidence'].mean():.1f}%")
    print(f"💾 Backup created: {backup_path}")
    print(f"📁 Enhanced export: test_results/enhanced_qualification_with_smykm_descriptions.csv")
    
    print(f"\n🎉 'SHOW ME YOU KNOW ME' ENHANCEMENT COMPLETE!")

if __name__ == "__main__":
    main()