#!/usr/bin/env python3
"""
INTEGRATE ENHANCED "SHOW ME YOU KNOW ME" DESCRIPTIONS
Update the qualification database with enhanced company blurbs and use case integrations
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
import os

class EnhancedDescriptionIntegrator:
    def __init__(self, db_path="data/fellow_qualification.db"):
        self.db_path = db_path
        self.enhanced_data = None
        
    def backup_database(self):
        """Create backup of current database"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = f"data/fellow_qualification_backup_{timestamp}.db"
        
        # Copy database file
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
        
    def update_database_schema(self):
        """Add new columns for enhanced descriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add enhanced description columns
        schema_updates = [
            "ALTER TABLE meetings ADD COLUMN enhanced_company_blurb TEXT",
            "ALTER TABLE meetings ADD COLUMN enhanced_use_case_integration TEXT", 
            "ALTER TABLE meetings ADD COLUMN enhancement_method TEXT DEFAULT 'show_me_you_know_me'",
            "ALTER TABLE meetings ADD COLUMN enhancement_confidence REAL",
            "ALTER TABLE meetings ADD COLUMN enhanced_at TIMESTAMP"
        ]
        
        for sql in schema_updates:
            try:
                cursor.execute(sql)
                print(f"✅ Schema update: {sql.split('ADD COLUMN')[1].split()[0] if 'ADD COLUMN' in sql else 'executed'}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e):
                    print(f"⚠️ Column already exists: {sql.split('ADD COLUMN')[1].split()[0] if 'ADD COLUMN' in sql else 'N/A'}")
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
        
    def update_meeting_records(self):
        """Update meeting records with enhanced descriptions"""
        if self.enhanced_data is None:
            raise ValueError("No enhanced data loaded. Call load_enhanced_data() first.")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updated_count = 0
        
        for _, row in self.enhanced_data.iterrows():
            fellow_meeting_id = row['fellow_meeting_id']
            
            # Update the meeting record
            update_sql = """
            UPDATE meetings 
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
                updated_count += 1
                print(f"✅ Updated: {row['company_name']} ({fellow_meeting_id})")
            else:
                print(f"⚠️ No record found for: {fellow_meeting_id}")
                
        conn.commit()
        conn.close()
        
        print(f"🎯 Total records updated: {updated_count}")
        return updated_count
        
    def generate_comparison_report(self, output_path="test_results/enhancement_comparison_report.csv"):
        """Generate before/after comparison report"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            company_name,
            fellow_meeting_id,
            company_blurb as original_blurb,
            enhanced_company_blurb,
            use_case as original_use_case,
            enhanced_use_case_integration,
            products_discussed as original_products,
            enhancement_confidence,
            enhanced_at
        FROM meetings 
        WHERE enhanced_company_blurb IS NOT NULL
        ORDER BY enhanced_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Calculate improvement metrics
        df['blurb_length_improvement'] = df['enhanced_company_blurb'].str.len() - df['original_blurb'].str.len()
        df['use_case_length_improvement'] = df['enhanced_use_case_integration'].str.len() - df['original_use_case'].fillna('').str.len()
        
        df.to_csv(output_path, index=False)
        print(f"✅ Comparison report generated: {output_path}")
        
        return df
        
    def export_enhanced_full_dataset(self, output_path="test_results/complete_enhanced_qualification_dataset.csv"):
        """Export complete dataset with enhanced descriptions"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            test_id,
            company_name,
            fellow_meeting_id,
            call_date,
            ae_name,
            industry,
            call_context,
            use_case,
            products_discussed,
            ae_next_steps,
            company_blurb,
            enhanced_company_blurb,
            enhanced_use_case_integration,
            company_age,
            employee_count,
            business_type,
            business_model,
            qualification_score,
            expected_routing,
            voice_ai_signals_count,
            business_signals_count,
            bi_extraction_confidence,
            enhancement_confidence,
            enhanced_at,
            processed_at
        FROM meetings 
        ORDER BY enhanced_at DESC, call_date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df.to_csv(output_path, index=False)
        print(f"✅ Complete enhanced dataset exported: {output_path}")
        print(f"📊 Total records: {len(df)}")
        
        return output_path

def main():
    """Main integration workflow"""
    print("🔧 ENHANCED DESCRIPTION DATABASE INTEGRATION")
    print("=" * 55)
    
    # Initialize integrator
    integrator = EnhancedDescriptionIntegrator()
    
    # Step 1: Backup database
    print("\n📦 Step 1: Creating database backup...")
    backup_path = integrator.backup_database()
    
    # Step 2: Update schema
    print("\n🏗️ Step 2: Updating database schema...")
    integrator.update_database_schema()
    
    # Step 3: Load enhanced data
    print("\n📊 Step 3: Loading enhanced descriptions...")
    # Find the most recent enhanced CSV
    import glob
    enhanced_files = glob.glob("test_results/show_me_you_know_me_enhanced_*.csv")
    if not enhanced_files:
        print("❌ No enhanced data files found!")
        return
        
    latest_file = max(enhanced_files, key=os.path.getctime)
    print(f"📁 Using latest file: {latest_file}")
    
    integrator.load_enhanced_data(latest_file)
    
    # Step 4: Update meeting records
    print("\n🔄 Step 4: Updating meeting records...")
    updated_count = integrator.update_meeting_records()
    
    # Step 5: Generate reports
    print("\n📈 Step 5: Generating comparison reports...")
    comparison_df = integrator.generate_comparison_report()
    
    # Step 6: Export complete dataset
    print("\n📤 Step 6: Exporting enhanced dataset...")
    full_dataset_path = integrator.export_enhanced_full_dataset()
    
    # Summary statistics
    print("\n📋 INTEGRATION SUMMARY")
    print("=" * 30)
    print(f"✅ Records enhanced: {updated_count}")
    print(f"📊 Average blurb length increase: {comparison_df['blurb_length_improvement'].mean():.0f} characters")
    print(f"🎯 Average confidence score: {comparison_df['enhancement_confidence'].mean():.1f}%")
    print(f"💾 Backup created: {backup_path}")
    print(f"📁 Enhanced dataset: {full_dataset_path}")
    
    # Display sample enhanced descriptions
    print("\n🎯 SAMPLE ENHANCED RESULTS")
    print("=" * 30)
    for _, row in comparison_df.head(2).iterrows():
        print(f"\n🏢 {row['company_name']}")
        print(f"📝 BEFORE: {row['original_blurb']}")
        print(f"✨ AFTER: {row['enhanced_company_blurb']}")
        print(f"🔧 INTEGRATION: {row['enhanced_use_case_integration'][:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()