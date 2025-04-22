import re
from openai import OpenAI
import os
import json

class AutoimmuneChecker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.path = "autoimmune_diseases.json"
        if os.path.exists(self.path):
            with open(self.path, "r") as inp:
                self.autoimmune_diseases = json.load(inp)
        else:
            self.autoimmune_diseases = {"yes":[],"no":[]}
        self.openai_client = None
    
    def is_autoimmune_disease(self, disease):
        if disease in self.autoimmune_diseases["yes"]:
            return True
        elif disease in self.autoimmune_diseases["no"]:
            return False
        
        yes_no_pattern = re.compile(r'^(yes|no)', re.IGNORECASE)
        yes_no_template = """
You must respond with a clear "Yes" or "No" first, followed by your explanation.
Your first word must be either "Yes" or "No".

Question: {question}
"""
        
        if not self.openai_client:
            self.openai_client = OpenAI(api_key=self.api_key)
        
        prompt = yes_no_template.format(question=f"Is '{disease}' an autoimmune disease?")
        
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives only Yes or No answers."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature = 0.1
        )
        
        response = completion.choices[0].message
        match = yes_no_pattern.match(response.content)
        
        if match:
            yes_no = match.group(1).lower() == 'yes'
        else:
            yes_no = False
            # throw?
            
        if yes_no:
            self.autoimmune_diseases["yes"].append(disease)
        else:
            self.autoimmune_diseases["no"].append(disease)
            
        return yes_no
    
    def save_diseases(self):
        with open(self.path, "w") as outp:
            json.dump(self.autoimmune_diseases, outp, indent=4)