# Day 23: LangChain + FAISS Local Vector Store Integration
# pip install faiss-cpu
# pip install langchain
# pip install openai
# pip install tiktoken
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# 1. Load your OpenAI API key (update path accordingly)
with open(r'D:\desktop\Key_GEN_AI.txt','r') as file:
    os.environ['OPENAI_API_KEY'] = file.read().strip()

# 2. Load documents (replace with your file path)
loader = TextLoader(r'D:\desktop\ML\NLP\speech.txt', encoding='utf-8')
documents = loader.load()

# 3 Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 4 Create embedding using OpenAI
embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=os.environ['OPENAI_API_KEY']
)

# 5 Create a FAISS vector store from documents and embeddings
faiss_index = FAISS.from_documents(docs, embeddings)

# 6 Setup retriever and LLM for QA
retriever = faiss_index.as_retriever(search_kwargs={"k":5})
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,
             openai_api_key=os.environ["OPENAI_API_KEY"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
                                       chain_type="stuff",
                                       return_source_documents=False
                                       )

# 7. Query the chain and get answers
def ask_question(question: str):
    response = qa_chain.invoke(question)
    return response['result']

if __name__ == "__main__":
    print("FAISS + Langchain QA DEMO\n")
    queries = [
        "What is the main topic of the speech?",
        "Who was addressing the Motion of No Confidence in the Lok Sabha?",
        "Which state did the Prime Minister assure of peace and development?",
        "Compare the PM’s portrayal of the opposition’s trust deficit with his government’s claimed achievements"
    ]
    for q in queries:
        print(f"Question:{q}")
        answer = ask_question(q)
        print(f"Answer: {answer}\n")

