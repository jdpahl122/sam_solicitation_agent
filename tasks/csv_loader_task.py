import csv
import os
from typing import List, Dict
from .base_task import BaseTask
from datetime import datetime, timezone


class CSVLoaderTask(BaseTask):
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        
    def execute(self) -> List[Dict]:
        """Load and process CSV data for embedding."""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
        opportunities = []
        
        # Try different encodings to handle various CSV file formats
        encodings_to_try = ['utf-8', 'utf-8-sig', 'windows-1252', 'iso-8859-1', 'cp1252']
        
        file_obj = None
        reader = None
        
        for encoding in encodings_to_try:
            try:
                file_obj = open(self.csv_file_path, 'r', encoding=encoding)
                reader = csv.DictReader(file_obj)
                # Test if we can read the first row
                first_row = next(reader)
                # Reset the file position to beginning
                file_obj.seek(0)
                reader = csv.DictReader(file_obj)
                print(f"✅ Successfully opened CSV file with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                if file_obj:
                    file_obj.close()
                continue
            except StopIteration:
                # Empty file
                if file_obj:
                    file_obj.close()
                raise ValueError("CSV file appears to be empty")
            except Exception as e:
                if file_obj:
                    file_obj.close()
                continue
        
        if not reader:
            raise ValueError(f"Could not read CSV file with any of these encodings: {encodings_to_try}")
        
        try:
            for row in reader:
                # Skip inactive opportunities
                if row.get('Active', '').lower() != 'yes':
                    continue
                    
                # Skip opportunities without valid response deadline
                response_deadline = row.get('ResponseDeadLine', '').strip()
                if not response_deadline or not self._is_future_deadline(response_deadline):
                    continue
                    
                # Clean and structure the data
                opportunity = {
                    'notice_id': row.get('NoticeId', '').strip(),
                    'title': row.get('Title', '').strip(),
                    'solicitation_number': row.get('Sol#', '').strip(),
                    'department': row.get('Department/Ind.Agency', '').strip(),
                    'office': row.get('Office', '').strip(),
                    'posted_date': row.get('PostedDate', '').strip(),
                    'notice_type': row.get('Type', '').strip(),
                    'base_type': row.get('BaseType', '').strip(),
                    'set_aside_code': row.get('SetASideCode', '').strip(),
                    'set_aside': row.get('SetASide', '').strip(),
                    'response_deadline': response_deadline,
                    'naics_code': row.get('NaicsCode', '').strip(),
                    'classification_code': row.get('ClassificationCode', '').strip(),
                    'location': self._format_location(row),
                    'award_amount': row.get('Award$', '').strip(),
                    'link': row.get('Link', '').strip(),
                    'description': row.get('Description', '').strip(),
                    'primary_contact_email': row.get('PrimaryContactEmail', '').strip(),
                    'primary_contact_name': row.get('PrimaryContactFullname', '').strip(),
                    'primary_contact_phone': row.get('PrimaryContactPhone', '').strip(),
                }
                
                # Only add if we have essential information
                if opportunity['notice_id'] and opportunity['title'] and opportunity['description']:
                    opportunities.append(opportunity)
                    
        finally:
            if file_obj:
                file_obj.close()
        
        print(f"✅ Loaded {len(opportunities)} active opportunities from CSV")
        return opportunities
    
    def _is_future_deadline(self, deadline_str: str) -> bool:
        """Check if the deadline is in the future."""
        try:
            # Handle different date formats
            for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d"]:
                try:
                    deadline = datetime.strptime(deadline_str.split('.')[0], fmt)
                    if not deadline.tzinfo:
                        deadline = deadline.replace(tzinfo=timezone.utc)
                    return deadline > datetime.now(timezone.utc)
                except ValueError:
                    continue
            return False
        except Exception:
            return False
    
    def _format_location(self, row: Dict) -> str:
        """Format location information from CSV row."""
        parts = []
        
        if row.get('PopCity'):
            parts.append(row['PopCity'].strip())
        if row.get('PopState'):
            parts.append(row['PopState'].strip())
        if row.get('PopZip'):
            parts.append(row['PopZip'].strip())
        if row.get('PopCountry'):
            parts.append(row['PopCountry'].strip())
            
        return ', '.join(parts) if parts else "" 
