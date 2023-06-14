import os
import streamlit as st
from langchain.chains import SequentialChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.9)


first_prompt = PromptTemplate(
    input_variables=["brand_name"],
    template="Tell me about {brand_name} product?",
)

# Memory
product_name_memory = ConversationBufferMemory(
    input_key='brand_name', memory_key='chat_history')
details_memory = ConversationBufferMemory(
    input_key='details', memory_key='chat_history')
suggestion_memory = ConversationBufferMemory(
    input_key='suggestions', memory_key='chat_history')

first_chain = LLMChain(llm=llm, prompt=first_prompt, verbose=True,
                       output_key='details', memory=product_name_memory)

second_prompt = PromptTemplate(
    input_variables=["details"],
    template="Suggest top 5 product brand like {details}",
)

second_chain = LLMChain(llm=llm, prompt=second_prompt, verbose=True,
                        output_key='suggestions', memory=product_name_memory)

third_prompt = PromptTemplate(
    input_variables=["suggestions"],
    template="Explain these {suggestions}",
)

third_chain = LLMChain(llm=llm, prompt=third_prompt, verbose=True,
                       output_key='description', memory=product_name_memory)

connected_chain = SequentialChain(
    chains=[first_chain, second_chain,third_chain], input_variables=['brand_name'], output_variables=['brand_name', 'details', 'suggestions'], verbose=True)


st.title('Search Brand Product Details')
input_text = st.text_input("Search a Product Name")


if input_text:
    st.write(connected_chain({'brand_name':input_text}))

    with st.expander('Product Details and Suggestions'): 
        st.info(product_name_memory.buffer)
