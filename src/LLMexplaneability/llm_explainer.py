from google import genai
import os

api_key= "AIzaSyCIOT_W5yg0s-Yan1A1StnHRftEl4OI4jk"

model="gemini-2.5-flash"


class LLMExplainer:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def explain_counterfactual(self, prompt):
        pass


class GeminiExplainer(LLMExplainer):

    def __init__(self, api_key, model):
        super().__init__(api_key, model)

    def explain_counterfactual(self, prompt):
        client = genai.Client(api_key = self.api_key)
        resp = client.models.generate_content(
            model = self.model,
            contents = prompt
        )
        return resp.text
    
    def export_explanation(self, explanation = None, prompt = None, file = "explanation.txt"):
        if explanation == None:
            explanation = self.explain_counterfactual(prompt)
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(explanation)          # sobrescribe si existe
    