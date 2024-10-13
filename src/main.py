from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import google.generativeai as genai
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
import cohere
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import redis
import json
import hashlib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Setup and initialization code
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("document")
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
gemini_model = genai.GenerativeModel(model_name='gemini-1.5-flash')
groq_model = ChatGroq(model_name="llama-3.2-11b-text-preview", temperature=0.3)

# Redis setup
redis_client = redis.Redis(
    host='get the host from upstash redis',
    port=6379,
    password='your-password',
    ssl=True
)

class ContextualRAG:
    def __init__(self):
        self.contextualized_chunks = []
        self.bm25 = None

    def preprocess(self, new_chunks: List[Dict[str, str]]):
        logger.info("Starting preprocessing...")
        new_contextualized_chunks = self.create_contextualized_chunks(new_chunks)
        self.contextualized_chunks.extend(new_contextualized_chunks)
        self.create_bm25_index()
        self.store_in_vector_database(new_contextualized_chunks)
        logger.info("Preprocessing complete.")

    def create_contextualized_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        logger.info("Creating contextualized chunks...")
        contextualized_chunks = []
        for chunk in chunks:
            cache_key = f"context:{hashlib.md5(chunk['content'].encode()).hexdigest()}"
            cached_context = redis_client.get(cache_key)
            if cached_context:
                context = cached_context.decode('utf-8')
            else:
                context = self.generate_chunk_context(chunk['content'])
                redis_client.set(cache_key, context.encode('utf-8'), ex=3600)  # Cache for 1 hour
            
            contextualized_chunks.append({
                "content": chunk['content'],
                "context": context,
                "source": chunk['source']
            })
        logger.info(f"Created {len(contextualized_chunks)} contextualized chunks.")
        return contextualized_chunks

    def generate_chunk_context(self, chunk_content: str) -> str:
        prompt = PromptTemplate.from_template(
            "Given the following text chunk, provide a brief context that situates this chunk within the overall document:\n\n{chunk_content}\n\nContext:"
        )
        response = groq_model.invoke(prompt.format(chunk_content=chunk_content))
        return response.content

    def create_bm25_index(self):
        logger.info("Creating BM25 index...")
        corpus = [f"{chunk['context']} {chunk['content']}" for chunk in self.contextualized_chunks]
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])
        logger.info("BM25 index created.")

    def store_in_vector_database(self, new_chunks: List[Dict[str, str]]):
        logger.info("Storing new contextualized chunks in vector database...")
        contextualized_texts = [
            f"{chunk['context']} {chunk['content']}" 
            for chunk in new_chunks
        ]
        embeddings = embedding_model.embed_documents(contextualized_texts)

        vectors = [
            (
                str(uuid.uuid4()),
                embedding,
                {
                    "content": chunk["content"],
                    "context": chunk["context"],
                    "source": chunk["source"]
                }
            )
            for chunk, embedding in zip(new_chunks, embeddings)
        ]

        index.upsert(vectors=vectors)
        logger.info(f"Stored {len(vectors)} new vectors in Pinecone.")

    def hybrid_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        cached_results = redis_client.get(cache_key)
        if cached_results:
            return json.loads(cached_results)

        logger.info(f"Performing hybrid search for query: {query}")
        query_embedding = embedding_model.embed_query(query)
        dense_results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
        
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_results = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k]
        
        combined_results = [
            {
                "content": match.metadata["content"],
                "context": match.metadata["context"],
                "source": match.metadata["source"],
                "score": match.score
            } for match in dense_results.matches
        ] + [
            {
                "content": self.contextualized_chunks[idx]["content"],
                "context": self.contextualized_chunks[idx]["context"],
                "source": self.contextualized_chunks[idx]["source"],
                "score": score
            } for idx, score in bm25_results
        ]
        
        seen = set()
        unique_results = []
        for result in combined_results:
            if result["content"] not in seen:
                seen.add(result["content"])
                unique_results.append(result)
        
        final_results = sorted(unique_results, key=lambda x: x["score"], reverse=True)[:k]
        redis_client.set(cache_key, json.dumps(final_results), ex=3600)  # Cache for 1 hour
        return final_results

    def rerank(self, query: str, results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        cache_key = f"rerank:{hashlib.md5((query + json.dumps(results)).encode()).hexdigest()}"
        cached_reranked = redis_client.get(cache_key)
        if cached_reranked:
            return json.loads(cached_reranked)

        logger.info("Reranking results...")
        rerank_response = cohere_client.rerank(
            model='rerank-english-v2.0',
            query=query,
            documents=[f"{r['context']} {r['content']}" for r in results],
            top_n=k
        )
        
        reranked = []
        for rerank_result in rerank_response.results:
            original = results[rerank_result.index]
            reranked.append({
                **original,
                "rerank_score": rerank_result.relevance_score
            })
        
        final_reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        redis_client.set(cache_key, json.dumps(final_reranked), ex=3600)  # Cache for 1 hour
        return final_reranked

    def generate_response(self, query: str, context: List[Dict[str, Any]], conversation_history: str) -> str:
        cache_key = f"response:{hashlib.md5((query + json.dumps(context) + conversation_history).encode()).hexdigest()}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            return cached_response.decode('utf-8')

        logger.info("Generating response...")
        context_str = "\n\n".join([f"Context: {c['context']}\nContent: {c['content']}" for c in context])
        prompt = f"""Given the following conversation history, context, and query, generate a comprehensive answer:

        Conversation History:
        {conversation_history}

        Context:
        {context_str}

        Query: {query}

        Answer:"""
        
        response = gemini_model.generate_content(prompt)
        final_response = response.text
        redis_client.set(cache_key, final_response.encode('utf-8'), ex=3600)  # Cache for 1 hour
        return final_response

    def answer_query_with_context(self, query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        search_results = self.hybrid_search(query)
        reranked_results = self.rerank(query, search_results)
        
        # Prepare conversation context
        context = "\n".join([f"Human: {turn['human']}\nAI: {turn['ai']}" for turn in conversation_history])
        context += f"\nHuman: {query}\nAI:"

        response = self.generate_response(query, reranked_results, context)
        
        return {
            "query": query,
            "response": response,
            "context": reranked_results[:3]  # Return top 3 context chunks
        }

rag_system = ContextualRAG()  # Initialize without chunks

@app.route('/process_document', methods=['POST'])
def process_document():
    logger.info("Received document processing request")
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            logger.info(f"Processing file: {file.filename}")
            file_content = file.read()
            pdf_reader = PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            
            new_chunks = [{"content": chunk, "source": file.filename} for chunk in chunks]
            rag_system.preprocess(new_chunks)
            
            logger.info("Document processed successfully")
            return jsonify({"message": "Document processed successfully"}), 200
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        logger.error("Invalid file format")
        return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    conversation_history = data.get('conversation_history', [])
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = rag_system.answer_query_with_context(query_text, conversation_history)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)