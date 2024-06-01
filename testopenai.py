# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
# Accessing the OPENAI KEY
import os
os.environ["OPENAI_API_KEY"] = ""
# Simple LLM call Using LangChain
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key="")
question = "what is the capital of France ?"
print(question, llm(question))

import torch
print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())