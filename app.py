import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st


llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.9)

prompt = PromptTemplate(
    input_variables=["product"],
    template="when the {product} brand created?",
)

chain = LLMChain(
    prompt=prompt,
    llm=llm,
)

st.title('Search Product Details')
input_text=st.text_input("Search a Product Name")

if input_text:
    st.write(chain.run(input_text))