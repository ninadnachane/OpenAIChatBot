import langchain
import faiss
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
from apikey import apikey
import os
import openai
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from typing import Any, List, Optional
from langchain.agents.agent import AgentExecutor


from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

os.environ['OPENAI_API_KEY'] = apikey
openai.api_key = os.environ['OPENAI_API_KEY']

st.subheader("Developed by Ninad Nachane")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vDB = FAISS.load_local("faiss_index/", openai_embeddings)
retriever = vDB.as_retriever(search_kwargs={"k": 6})
tool = create_retriever_tool(
    retriever,
    "Query_PBI_Model",
    "useful to fetch data from Power BI data model."
    )
tools = [tool]

llm = ChatOpenAI(temperature=0,model="gpt-4")
llm_retrieval_agent = create_conversational_retrieval_agent(llm=llm,tools=tools, verbose=True)
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
prompt = OpenAIFunctionsAgent.create_prompt(
        # system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                   return_intermediate_steps=True)
# result = agent_executor("hi, Generate a summary for all the expenses for country France for month of Jan 2023. Print total as well")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

st.title("Power BI Chat-Agent")
...
response_container = st.container()
textcontainer = st.container()


conversation = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.buffer_memory, verbose=True)
                                #    return_intermediate_steps=True)
# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            response = conversation(query)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response['output'])

with response_container:
     response['output']
    # if st.session_state['responses']:
    #     for i in range(len(st.session_state['responses'])):
    #         message(st.session_state['responses'])
            # if i < len(st.session_state['requests']):
            #     message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
