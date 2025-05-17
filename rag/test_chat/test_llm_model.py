from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from config import *
import os
from langchain_community.chat_models import ChatOllama
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def get_llm(model_id="bert-base-cased", device=0, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    return hf

def get_llm_gemini(temperature=0, max_tokens=None, timeout=None, max_retries=2):
    # initialize llm 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    return llm

def get_api_llm(model_name: str="llama3.2"):

    llm = ChatOllama(model=model_name, max_tokens=512, temperature=0)

    return llm


