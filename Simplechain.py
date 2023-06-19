import os
from dotenv import load_dotenv, find_dotenv
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
#from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

_=load_dotenv(find_dotenv())
openai.api_key=os.getenv('OPENAI_API_KEY')

st.title('Search your Topic')
input_text=st.text_input('Enter your topic')

l=OpenAI(temperature=0.8)


first_input=PromptTemplate(input_variables=['name'],
                           template="Tell me about {name}")



person_memory=ConversationBufferMemory(input_key='name',memory_key='person_history')

chain=LLMChain(llm=l,prompt=first_input,verbose=True,output_key='person',memory=person_memory)

second_input=PromptTemplate(input_variables=['person'],
                            template="When was {person} born")
dob_memory=ConversationBufferMemory(input_key='person',memory_key='dob_history')

chain2=LLMChain(llm=l,prompt=second_input,output_key='dob',verbose=True, memory=dob_memory)

third_input=PromptTemplate(input_variables=['dob'],
                            template="Mention five major event in this {dob}")
desc_memory=ConversationBufferMemory(input_key='dob',memory_key='desc_history')

chain3=LLMChain(llm=l,prompt=third_input,output_key='description',verbose=True, memory=desc_memory)



#parent_chain=SimpleSequentialChain(chains=[chain,chain2],verbose=True)


parent_chain=SequentialChain(chains=[chain,chain2,chain3],verbose=True,input_variables=['name'],output_variables=['person','dob','description'])

if input_text:
    st.write(parent_chain(input_text))
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(desc_memory.buffer)
   

