from typing import List, Dict, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from .base_chain import BaseChain
from utils.prompt_loader import load_prompt


class OpportunityMatchingChain(BaseChain):
    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(model=model_name)
        self.prompt_template = load_prompt("opportunity_evaluation_prompt.txt")
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.output_parser = StrOutputParser()
        
        # Create the evaluation chain
        self.evaluation_chain = self.prompt | self.llm | self.output_parser
    
    def evaluate_opportunity(self, opportunity: Dict, company_profile: str) -> Dict:
        """Evaluate a single opportunity against the company profile."""
        
        # Extract metadata for the prompt
        metadata = opportunity.get('metadata', {})
        description = opportunity.get('text', '')
        
        # Clean up the description to extract just the description part
        if 'Description: ' in description:
            description = description.split('Description: ', 1)[1].strip()
        
        # Create the evaluation
        evaluation = self.evaluation_chain.invoke({
            'company_profile': company_profile,
            'title': metadata.get('title', ''),
            'department': metadata.get('department', ''),
            'office': metadata.get('office', ''),
            'set_aside': metadata.get('set_aside', ''),
            'naics_code': metadata.get('naics_code', ''),
            'classification_code': metadata.get('classification_code', ''),
            'location': metadata.get('location', ''),
            'award_amount': metadata.get('award_amount', ''),
            'response_deadline': metadata.get('response_deadline', ''),
            'solicitation_number': metadata.get('solicitation_number', ''),
            'description': description
        })
        
        # Parse the match score from the evaluation
        match_score = self._extract_match_score(evaluation)
        
        return {
            'opportunity': opportunity,
            'evaluation': evaluation,
            'match_score': match_score,
            'metadata': metadata
        }
    
    def rank_opportunities(self, opportunities: List[Dict], company_profile: str, top_k: int = 10) -> List[Dict]:
        """Rank opportunities based on how well they match the company profile."""
        
        print(f"🔍 Evaluating {len(opportunities)} opportunities against company profile...")
        
        evaluated_opportunities = []
        
        for i, opportunity in enumerate(opportunities):
            if i % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(opportunities)} opportunities evaluated")
            
            try:
                result = self.evaluate_opportunity(opportunity, company_profile)
                evaluated_opportunities.append(result)
            except Exception as e:
                print(f"⚠️ Error evaluating opportunity {opportunity.get('metadata', {}).get('title', 'Unknown')}: {e}")
                continue
        
        # Sort by match score (highest first)
        evaluated_opportunities.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Return top k opportunities
        return evaluated_opportunities[:top_k]
    
    def _extract_match_score(self, evaluation: str) -> int:
        """Extract the match score from the evaluation text."""
        # Look for "MATCH SCORE: [number]" pattern
        match = re.search(r'MATCH SCORE:\s*(\d+)', evaluation, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: look for any score pattern
        match = re.search(r'(\d+)/100', evaluation)
        if match:
            return int(match.group(1))
        
        # Default to 0 if no score found
        return 0
    
    def execute(self, opportunities: List[Dict], company_profile: str, top_k: int = 10) -> List[Dict]:
        """Main execution method for the chain."""
        return self.rank_opportunities(opportunities, company_profile, top_k) 
