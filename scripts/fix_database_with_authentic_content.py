#!/usr/bin/env python3
"""
FIX DATABASE WITH AUTHENTIC CONTENT
Replace AI-generated garbage with REAL call content
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

class DatabaseAuthenticFixer:
    def __init__(self, db_path="data/fellow_qualification.db"):
        self.db_path = db_path
        self.authentic_data = None
        
    def backup_database(self):
        """Create backup before fixing"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = f"data/fellow_qualification_backup_before_authentic_fix_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
        
    def load_authentic_data(self, csv_path):
        """Load authentic call content"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Authentic data file not found: {csv_path}")
            
        self.authentic_data = pd.read_csv(csv_path)
        print(f"✅ Loaded authentic call data: {len(self.authentic_data)} companies")
        return self.authentic_data
        
    def clear_ai_garbage(self):
        """Remove AI-generated content and replace with authentic call content"""
        if self.authentic_data is None:
            raise ValueError("No authentic data loaded. Call load_authentic_data() first.")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        fixed_count = 0
        
        for _, row in self.authentic_data.iterrows():
            fellow_meeting_id = row['fellow_meeting_id']
            
            # Update with REAL call content
            update_sql = """
            UPDATE qualification_logs 
            SET enhanced_company_blurb = ?,
                enhanced_use_case_integration = ?,
                enhancement_method = 'authentic_call_content_only',
                enhancement_confidence = 100.0,
                enhanced_at = ?
            WHERE fellow_meeting_id = ?
            """
            
            cursor.execute(update_sql, (
                row['actual_company_description'],
                row['actual_use_case_discussed'], 
                datetime.now().isoformat(),
                fellow_meeting_id
            ))
            
            if cursor.rowcount > 0:
                fixed_count += cursor.rowcount
                print(f"✅ FIXED {cursor.rowcount} records for: {row['company_name']} ({fellow_meeting_id})")
                print(f"   Old AI garbage replaced with: {row['actual_company_description'][:60]}...")
            else:
                print(f"⚠️ No records found for: {fellow_meeting_id}")
                
        conn.commit()
        conn.close()
        
        print(f"🎯 Total records FIXED: {fixed_count}")
        return fixed_count
        
    def generate_authentic_comparison_report(self, output_path="test_results/AUTHENTIC_VS_AI_GARBAGE_COMPARISON.csv"):
        """Generate comparison showing what was AI garbage vs authentic"""
        
        # Load the AI-enhanced data we created earlier
        ai_enhanced_files = [f for f in os.listdir("test_results") if f.startswith("show_me_you_know_me_enhanced_")]
        if not ai_enhanced_files:
            print("⚠️ No AI-enhanced files found for comparison")
            return None
            
        latest_ai_file = f"test_results/{max(ai_enhanced_files)}"
        ai_df = pd.read_csv(latest_ai_file)
        
        # Create comparison
        comparison_data = []
        
        for _, auth_row in self.authentic_data.iterrows():
            fellow_id = auth_row['fellow_meeting_id']
            ai_row = ai_df[ai_df['fellow_meeting_id'] == fellow_id].iloc[0] if len(ai_df[ai_df['fellow_meeting_id'] == fellow_id]) > 0 else None
            
            comparison = {
                'company_name': auth_row['company_name'],
                'fellow_meeting_id': fellow_id,
                
                # AUTHENTIC CONTENT (what was actually discussed)
                'AUTHENTIC_company_description': auth_row['actual_company_description'],
                'AUTHENTIC_use_case': auth_row['actual_use_case_discussed'],
                'AUTHENTIC_products': auth_row['actual_products_mentioned'],
                
                # AI GARBAGE (what I made up)
                'AI_GARBAGE_company_blurb': ai_row['enhanced_company_blurb'] if ai_row is not None else 'N/A',
                'AI_GARBAGE_integration': ai_row['enhanced_use_case_integration'] if ai_row is not None else 'N/A',
                
                # Source verification
                'source_transcript_snippet': auth_row['transcript_snippet'],
                'source_call_notes': auth_row['call_notes'],
                
                'fixed_at': datetime.now().isoformat()
            }
            comparison_data.append(comparison)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_path, index=False)
        
        print(f"✅ Authentic vs AI garbage comparison: {output_path}")
        return comparison_df
        
    def export_fixed_database(self, output_path="test_results/AUTHENTIC_QUALIFICATION_DATA_FIXED.csv"):
        """Export database with authentic content only"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT DISTINCT
            fellow_meeting_id,
            company_name,
            enhanced_company_blurb as authentic_company_description,
            enhanced_use_case_integration as authentic_use_case_discussed,
            enhancement_method,
            enhancement_confidence,
            enhanced_at as fixed_at
        FROM qualification_logs 
        WHERE enhancement_method = 'authentic_call_content_only'
        GROUP BY fellow_meeting_id, company_name
        ORDER BY fixed_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df.to_csv(output_path, index=False)
        print(f"✅ Fixed authentic database export: {output_path}")
        print(f"📊 Records exported: {len(df)}")
        
        return df

def main():
    """Fix the database with authentic content"""
    print("🚨 FIXING DATABASE - REMOVING AI GARBAGE, ADDING AUTHENTIC CONTENT")
    print("=" * 70)
    print("🎯 Replacing made-up enhancements with REAL call content")
    print()
    
    # Initialize fixer
    fixer = DatabaseAuthenticFixer()
    
    # Step 1: Backup database
    print("📦 Step 1: Creating backup before fix...")
    backup_path = fixer.backup_database()
    
    # Step 2: Load authentic data
    print("\n📊 Step 2: Loading authentic call content...")
    import glob
    authentic_files = glob.glob("test_results/AUTHENTIC_CALL_EXTRACTION_*.csv")
    if not authentic_files:
        print("❌ No authentic extraction files found!")
        return
        
    latest_authentic_file = max(authentic_files, key=os.path.getctime)
    print(f"📁 Using authentic file: {latest_authentic_file}")
    
    fixer.load_authentic_data(latest_authentic_file)
    
    # Step 3: Fix database
    print("\n🔧 Step 3: Replacing AI garbage with authentic call content...")
    fixed_count = fixer.clear_ai_garbage()
    
    # Step 4: Generate comparison report
    print("\n📋 Step 4: Generating authentic vs AI garbage comparison...")
    comparison_df = fixer.generate_authentic_comparison_report()
    
    # Step 5: Export fixed database
    print("\n📤 Step 5: Exporting fixed authentic database...")
    fixed_df = fixer.export_fixed_database()
    
    # Display authentic content
    print("\n🎯 AUTHENTIC CALL CONTENT (What was REALLY discussed):")
    print("=" * 60)
    
    for _, row in fixed_df.iterrows():
        print(f"\n🏢 {row['company_name']}")
        print(f"📞 Fellow ID: {row['fellow_meeting_id']}")
        print(f"\n✅ AUTHENTIC Company Description (from actual call):")
        print(f"   {row['authentic_company_description']}")
        print(f"\n✅ AUTHENTIC Use Case (from actual call):")  
        print(f"   {row['authentic_use_case_discussed']}")
        print("-" * 60)
    
    # Summary
    print(f"\n📊 FIX COMPLETE SUMMARY")
    print("=" * 35)
    print(f"✅ Database records fixed: {fixed_count}")
    print(f"🏢 Companies corrected: {len(fixed_df)}")
    print(f"🎯 Method: Authentic call content only")
    print(f"💾 Backup created: {backup_path}")
    print(f"📁 Fixed database export: test_results/AUTHENTIC_QUALIFICATION_DATA_FIXED.csv")
    print(f"📊 Comparison report: test_results/AUTHENTIC_VS_AI_GARBAGE_COMPARISON.csv")
    
    print(f"\n🎉 DATABASE FIXED WITH AUTHENTIC CONTENT!")
    print(f"🚫 AI-generated garbage removed")
    print(f"✅ Real call content restored")

if __name__ == "__main__":
    main()