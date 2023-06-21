import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import FewShotPromptTemplate

_=load_dotenv(find_dotenv())
openai.api_key=os.getenv('OPENAI_API_KEY')

demo_tem=''' I want you to act as a financial advisor for people. In an easy way, explain the basics of {financial_concept}.'''
prompt=PromptTemplate(input_variables=['financial_concept'],
                      template=demo_tem)
#print(prompt.format(financial_concept='income tax'))

llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=prompt)
print(chain.run('GDP'))

print("******Language Translation**********")

template_1='''Translate the following sentence {sentence} into {target_language}'''
lang_prompt=PromptTemplate(input_variables=['sentence','target_language'],template=template_1)
chain2=LLMChain(llm=llm, prompt=lang_prompt)
print(chain2.run({'sentence':'Hellow, How r u','target_language':'Telugu'}))


print("*******FewShotPromptTemplate*******")

examples=[{'word':'tall','antonym':'short'},
{'word':'happy','antonym':'sad'}]

format_template="""word:{word}, antonym:{antonym}"""

example_prompt_f=PromptTemplate(input_variables=["word","antonym"],template=format_template)

few_shot=FewShotPromptTemplate(examples=examples,example_prompt=example_prompt_f,prefix="Given the antonym of every input\n",suffix="word:{input}\n antonym:",input_variables=["input"],example_separator="\n",)

print(few_shot.format(input='big'))

chain3=LLMChain(llm=llm, prompt=few_shot)
print(chain3({'input':"big"}))
print(chain3.run('big'))