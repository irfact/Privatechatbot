# Privatechatbot
Chatbot Assistant that can read personal files.LANGCHAIN/PINECONE/STREAMLIT
**New features and their descriptions**
This Chatbot is a pesonalized assistant which can read any unstructured text files (.csv,.pdf,.txt,.docx) and process the information.
the processed information will be stored in avector database for which we have chosen PINECONE. a scalable high performance lets you deliver remarkable GenAI applications faster, at up to 50x lower cost.
The OPEN AI model currently used for this project is GPT turbo 3.5 turbo which can understand and generate natural language or code and have been optimized for chat using the Chat Completions API but work well for non-chat tasks as well.
Specifications:
MODEL->	DESCRIPTION->CONTEXT WINDOW->TRAINING DATA
gpt-3.5-turbo-0125	New Updated GPT 3.5 Turbo
The latest GPT-3.5 Turbo model with higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Returns a maximum of 4,096 output tokens..	16,385 tokens	Up to Sep 2021
and embedding model is text-embedding-3-small


**#Installation**
Create a virtual environment to install python. for this project. i used python 3.11
More details to install python,virtual environment to store your packages and to perform your first API request with OpenAI model can be found here

https://platform.openai.com/docs/quickstart?context=python

and run the requirement.txt file using the below command

pip install -r requirements.txt

After the above code is run and successfully installed
Navigate to the folder where the script is placed and 
create a subfolder {.streamlit} and copy an paste the secrets.toml file from the git maintree
![image](https://github.com/irfact/Privatechatbot/assets/60041978/5945bdd7-b452-4335-b3c0-4eccba1605e9)

Also pull the subfolder BGimages into the project
run the below command in the terminal or command prompt

streamlit run {yourpythonscriptname.py}
(eg: streamlit run pathscape.py)

**MODEL FRAMEWORK AND METHODS**
This Chatbot is designed on Retrieval-Augmented Generation, aka RAG
Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
Langchain Many Large Language Models applications require user-specific data that is not part of the model's training set.
LangChain provides all the building blocks for RAG applications
For our Chatbot we used Langchain for
Document Loading
Document splitting
Embedding to Vector
Conversation chain
Conversation buffer memory

Pinecone is a vector database for our chatbot
OPEN AI is the embedding and generative model used (text-embedding-3-small and 3.5-gpt-turbo-0125) # can be configurable in the secrets.toml

For FrontEnd design STREAMLIT(Streamlit turns data scripts into shareable web apps in minutes.All in pure Python.)
https://blog.streamlit.io/generative-ai-and-streamlit-a-perfect-match/


The Architecture of the Model is Diagramatically visualized below:

![image](https://github.com/irfact/Privatechatbot/assets/60041978/bdfb8694-c768-4917-ab1b-570b9df58ef4)


