# how to create the virtual env:
#  step 1 python -m venv .venv
# step 2 .venv\Scripts\Activate.ps1
# pip install pinecone-client
# pip install --upgrade pinecone
# pip show pinecone
# pip install -U langchain-community
# pip install -U langchain-pinecone
# pip install -U langchain langchain-community langchain-openai langchain-pinecone pinecone
# pip install langchain
# pip install openai
# pip install langchain-openai
# Go to https://www.pinecone.io
# git init
# git add .
# git commit -m "Add Day 22 pinecone with langchain"
# git remote add origin https://github.com/SharmaPankaj3/Day-22_Pinecone-Integration-with-LangChain-.git
# git branch -M main

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# 1. Load API Key
with open(r'D:\desktop\Key_GEN_AI.txt','r') as f:
    os.environ['OPENAI_API_KEY'] = f.read().strip()
# Load Pine API Key
with open(r'D:\desktop\pinecone_key.txt','r') as file:
    os.environ['PINECONE_API_KEY'] = file.read().strip()

# 2. Initialize Pinecone
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    INDEX_NAME = "langchain-demo-index"
    # Create index only if it doesn't exist
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # matches ada-002 embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created index: {INDEX_NAME}")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")
# 3. Load and split documents
loader = TextLoader(r"D:\desktop\ML\NLP\speech.txt",encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)
# 4. Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ['OPENAI_API_KEY'])

# 5. Upload documents to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    docs,
    embedding=embeddings,
    index_name=INDEX_NAME
)

# ==== 7. RetrievalQA ====
llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
# ==== 8. Ask a Question ====
query = "What is the main topic of the speech?"
response = qa.invoke(query)
print("Answer:", response)

query_1 = "Who was addressing the Motion of No Confidence in the Lok Sabha?"
response = qa.invoke(query_1)
print("Answer:", response)

query_2= "Which state did the Prime Minister assure of peace and development?"
response = qa.invoke(query_2)
print("Answer:", response)

query_3= "Compare the PM’s portrayal of the opposition’s trust deficit with his government’s claimed achievements"
response = qa.invoke(query_3)
print("Answer:", response)


# How many people did the NITI report say came out of poverty?

# What did the IMF working paper state about extreme poverty in India?

# Which infrastructure developments in the Northeast were mentioned?

# What is the government’s mantra mentioned for economic growth?

# Which two “I”s did the PM say stand for arrogance in the opposition alliance?

# What examples did the PM give of first-time achievements in the Northeast?
