from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone,PodSpec
import time
import openai

##chatbot packages
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

api = st.sidebar.text_input("API-KEY", type ="password")
directory = st.sidebar.text_input("Enter your file path")


def load_docs(directory):

   loader = DirectoryLoader(directory)
   documents = loader.load()

   return documents

documents = load_docs(directory)
len(documents)

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)

  return docs

docs = split_docs(documents)
print(len(docs))


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key = api)

import os

use_serverless = False
pc= Pinecone(
    api_key = os.environ['PINECONE_API_KEY']) # find at app.pinecone.io

# check for and delete index if already exists
index_name = 'pathscapechatbot'

# create a new index
if not index_name:
 pc.create_index(
    index_name,
    dimension=1536,  # dimensionality of text-embedding-ada-002
    metric='cosine',
    spec=PodSpec(
        environment="gcp-starter",
        pod_type="starter",
        pods=1
    )
)
# next to api key in console



while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)


index = pc.Index(index_name)
index.describe_index_stats()
from langchain.vectorstores import Pinecone as pineconestore
index = pineconestore.from_documents(docs, embeddings, index_name=index_name )


st.subheader("Pathscape Chatbot with Pinecone,Langchain and ADA-002")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=8,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api)
...
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


st.title("Pathscape Chatbot")
#container for chat history
response_container = st.container()
#container for textbox
textcontainer = st.container()


def query_refiner(conversation, query):
        openai.api_key = api

        prompt = f"You are a helpful assistant that generates refined questions based on a topic. Respond with one short question with the conversation and query asked in the textbox.\n\nCONVERSATION: {conversation}\n\nQUERY: {query}\n\nRefined Question:"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "provide a refined query"},
                {"role": "assistant", "content": "you are a helpful assistant which refines my query"}
            ],

        )

        return response.choices[0]["message"]["content"]


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string


def find_match(input):
    result= index.similarity_search(input,k=6)

    st.write(result)
    return result
    #return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string,query)
            st.subheader("refined query:")
            st.write(refined_query)
            context = find_match(query)


            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
    ...
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')








