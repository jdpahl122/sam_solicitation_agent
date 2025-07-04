from typing import List, Dict
from .base_task import BaseTask


class CSVPreprocessTask(BaseTask):
    def execute(self, opportunities: List[Dict]) -> List[Dict]:
        """Process CSV opportunities for embedding."""
        processed_docs = []
        
        for opp in opportunities:
            # Create a comprehensive text representation for embedding
            text_parts = []
            
            # Title and description are most important
            if opp.get('title'):
                text_parts.append(f"Title: {opp['title']}")
            
            if opp.get('description'):
                text_parts.append(f"Description: {opp['description']}")
            
            # Add key details
            if opp.get('department'):
                text_parts.append(f"Department: {opp['department']}")
            
            if opp.get('office'):
                text_parts.append(f"Office: {opp['office']}")
            
            if opp.get('set_aside'):
                text_parts.append(f"Set-Aside: {opp['set_aside']}")
            
            if opp.get('naics_code'):
                text_parts.append(f"NAICS Code: {opp['naics_code']}")
            
            if opp.get('classification_code'):
                text_parts.append(f"Classification: {opp['classification_code']}")
            
            if opp.get('location'):
                text_parts.append(f"Location: {opp['location']}")
            
            if opp.get('award_amount'):
                text_parts.append(f"Award Amount: {opp['award_amount']}")
            
            # Join all parts
            text = "\n".join(text_parts)
            
            # Create metadata for filtering and display
            metadata = {
                "notice_id": opp.get('notice_id') or "",
                "title": opp.get('title') or "",
                "solicitation_number": opp.get('solicitation_number') or "",
                "department": opp.get('department') or "",
                "office": opp.get('office') or "",
                "posted_date": opp.get('posted_date') or "",
                "notice_type": opp.get('notice_type') or "",
                "base_type": opp.get('base_type') or "",
                "set_aside_code": opp.get('set_aside_code') or "",
                "set_aside": opp.get('set_aside') or "",
                "response_deadline": opp.get('response_deadline') or "",
                "naics_code": opp.get('naics_code') or "",
                "classification_code": opp.get('classification_code') or "",
                "location": opp.get('location') or "",
                "award_amount": opp.get('award_amount') or "",
                "link": opp.get('link') or "",
                "primary_contact_email": opp.get('primary_contact_email') or "",
                "primary_contact_name": opp.get('primary_contact_name') or "",
                "primary_contact_phone": opp.get('primary_contact_phone') or "",
            }
            
            processed_docs.append({
                "text": text,
                "metadata": metadata
            })
        
        print(f"✅ Preprocessed {len(processed_docs)} CSV opportunities for embedding")
        return processed_docs 
