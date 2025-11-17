from google import genai
# import ollama
import os
import time
import random as rd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLMExplainer:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def explain_counterfactual(self, prompt):
        pass

    def export_explanation(self, explanation = None, prompt = None, file = "explanation.txt"):
        if explanation == None:
            explanation = self.explain_counterfactual(prompt)
        
        with open(file, "w", encoding="utf-8") as f:
            f.write(explanation)          # sobrescribe si existe
    



class GeminiExplainer(LLMExplainer):

    def __init__(self):
        api_key= "AIzaSyCIOT_W5yg0s-Yan1A1StnHRftEl4OI4jk"
        model = "gemini-2.5-pro"
        super().__init__(api_key, model)

    def explain_counterfactual(self, prompt):

        # time.sleep(seconds) # wait for 60 seconds before retrying
        client = genai.Client()

        num_tries = 0
        # while num_tries < 30:
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return resp.text
        except Exception as e:
            num_tries += 1
            # time.sleep(60) # wait for 60 seconds before retrying
            
        # raise Exception("Failed to get response from the model after multiple attempts.")
    

  
class LocalLlamaExplainer(LLMExplainer):

    def __init__(self):
        api_key= None
        model = "Llama-3.1-8B"
        self.path = os.path.abspath(os.path.join('..', ''))
        super().__init__(api_key, model)

    def explain_counterfactual(self, prompt):

        local_model_path = os.path.join(self.path, self.model)

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)

        # Mover a GPU si está disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)


        # Tokenizar
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        num_tries = 0
        while num_tries < 30:
            try:
                # Generar texto
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,  # cuántos tokens generar
                    temperature=0.7,     # creatividad
                    do_sample=True,      # True para sampling
                    top_p=0.9,           # nucleus sampling
                    repetition_penalty=1.2
                )

                # Decodificar y mostrar
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return resp
            except Exception as e:
                num_tries += 1
                time.sleep(60) # wait for 60 seconds before retrying
            
        raise Exception("Failed to get response from the model after multiple attempts.")

# class OllamaExplainer(LLMExplainer):

#     def __init__(self):
#         api_key= None
#         model = "llama3"
#         super().__init__(api_key, model)

#     def explain_counterfactual(self, prompt):

#         seconds = rd.randint(1, 20)
#         # time.sleep(seconds) # wait for 60 seconds before retrying
#         client = ollama.Client()


#         num_tries = 0
#         # while num_tries < 30:
#         try:
#             resp = client.generate(
#                 model=self.model,
#                 prompt=prompt,
#                 stream=False # Configuramos stream en False para obtener la respuesta completa de una vez
#             )
            
#             return resp.response
#         except Exception as e:
#             num_tries += 1
#             # time.sleep(60) # wait for 60 seconds before retrying
            
#         # raise Exception("Failed to get response from the model after multiple attempts.")
