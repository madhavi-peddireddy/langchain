import os
import streamlit as st
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv


# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')


st.title('Search Your Topic :mag: ')
input_text=st.text_input('Enter your topic')

llm = OpenAI(temperature=0.8)


if input_text:
    st.write(llm(input_text))



