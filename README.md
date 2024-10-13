# Contextual RAG Agent Project

## Overview

This project implements a Contextual Retrieval-Augmented Generation (RAG) system using Flask for the backend, Streamlit for the frontend, and various AI services. The system allows users to upload PDF documents, process them, and then query the processed information using natural language.

## Repository Structure

```
CONTEXTUAL_RAG_AGENT/
├── .env                  # Environment variables (not in repo)
├── app.py                # Streamlit frontend application
├── dockerfile            # Docker configuration for backend
├── main.py               # Flask backend application
├── requirements.txt      # Python dependencies
└── Contextual_RAG.ipynb  # Google Colab notebook with step-by-step implementation
```

## Components

1. **Flask Backend** (`main.py`): Handles document processing, querying, and interfacing with AI services.
2. **Streamlit Frontend** (`app.py`): Provides a user-friendly interface for interacting with the system.
3. **Docker**: Containerizes the backend application for easy deployment and scalability.
4. **Google Colab Notebook** (`Contextual_RAG.ipynb`): Contains the step-by-step implementation and explanation of the RAG system.
5. **AI Services**: Utilizes Pinecone, Cohere, Google AI, and Groq for various AI tasks.

## Prerequisites

- Docker
- Python 3.11
- API keys for Pinecone, Cohere, Google AI, and Groq
- Streamlit

## Setup

1. **Clone the Repository**
   ```
   git clone https://github.com/Pandurangmopgar/Contextual_Retrival.git
   cd CONTEXTUAL_RAG_AGENT
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory with the following content:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   COHERE_API_KEY=your_cohere_api_key
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```

4. **Dockerfile**
   The Dockerfile is set up to create a container for the Flask backend:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY main.py .
   COPY .env .
   EXPOSE 5000
   ENV FLASK_APP=main.py
   ENV FLASK_RUN_HOST=0.0.0.0
   CMD ["flask", "run", "--host=0.0.0.0"]
   ```

## Running the Application

1. **Build and Run the Backend Docker Container**
   ```
   docker build -t contextual-rag-backend .
   docker run -p 5000:5000 contextual-rag-backend
   ```

2. **Run the Streamlit Frontend**
   ```
   streamlit run app.py
   ```

The backend will be accessible at `http://localhost:5000`, and the Streamlit frontend will open in your default web browser.

## Google Colab Notebook

The `Contextual_RAG.ipynb` notebook in the repository provides a detailed, step-by-step implementation of the RAG system. To use it:

1. Open the notebook in Google Colab.
2. Follow the instructions to set up your environment and API keys.
3. Run through the cells to understand the implementation details and experiment with the system.

## Usage

1. **Process a Document**
   - Use the Streamlit interface to upload a PDF file.
   - The system will process and store the document information.

2. **Query the System**
   - Enter your query in the Streamlit interface.
   - The system will retrieve relevant information from processed documents.

## System Architecture

The Contextual RAG system works as follows:

1. **Document Processing**:
   - PDFs are uploaded and text is extracted.
   - Text is split into chunks.
   - Each chunk is contextualized using AI.
   - Contextualized chunks are stored in Pinecone and a local BM25 index.

2. **Querying**:
   - User sends a query through the Streamlit interface.
   - Backend performs hybrid search (vector + BM25).
   - Results are re-ranked.
   - Final answer is generated using retrieved context.

3. **Caching**:
   - Redis is used to cache various results to improve performance.

## Troubleshooting

- Ensure all API keys are correctly set in the `.env` file.
- Check Docker logs for backend errors:
  ```
  docker logs [container_id]
  ```
- For frontend issues, check the Streamlit console output.
- Verify that all required packages are installed and listed in `requirements.txt`.

## Contributing

[Add your guidelines for contributing to the project]

## License

[Specify the license under which this project is released]