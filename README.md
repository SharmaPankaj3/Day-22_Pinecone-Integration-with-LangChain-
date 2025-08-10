# Day 22 – RAG with Pinecone

## Overview
This script implements Retrieval-Augmented Generation using LangChain, Pinecone, and OpenAI.
We load a `.txt` speech file, split it into chunks, generate embeddings, store them in Pinecone, and query them via RetrievalQA.

## Features
- Vector DB storage with Pinecone
- Semantic search on custom documents
- OpenAI embeddings (`text-embedding-ada-002`)
- RetrievalQA for contextual answers

## Usage
1. Set `OPENAI_API_KEY` in `Key_GEN_AI.txt`
2. Set `PINECONE_API_KEY` in `pinecone_key.txt`
3. Place your text file (e.g., `speech.txt`) in the given path.
4. Install dependencies:
# pip install pinecone-client
# pip install --upgrade pinecone
# pip show pinecone
# pip install -U langchain-community
# pip install -U langchain-pinecone
# pip install -U langchain langchain-community langchain-openai langchain-pinecone pinecone
# pip install langchain
# pip install openai
# pip install langchain-openai
 Go to https://www.pinecone.io
# git init
# git add .
# git commit -m "Add Day 22 pinecone with langchain"
# git remote add origin https://github.com/SharmaPankaj3/Day-22_Pinecone-Integration-with-LangChain-.git
# git branch -M main
# git push -u origin main
