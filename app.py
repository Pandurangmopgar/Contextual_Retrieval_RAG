import streamlit as st
import requests
from typing import Dict, Any
import re

# Set page config at the very beginning
st.set_page_config(page_title="Contextual RAG Assistant", page_icon="🧠", layout="wide")

# Set the backend URL
BACKEND_URL = "http://localhost:5000"  # Adjust if your backend is hosted elsewhere

# Custom CSS for modern dark theme with blue gradient card
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .main-card {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin: -1rem;
    }
    .sidebar .sidebar-content > div {
        background-color: #3D3D3D;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .stTextInput > div > div > input {
        background-color: #3D3D3D;
        color: #FFFFFF;
        border: 1px solid #4B6CB7;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: #4B6CB7;
        color: #FFFFFF;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #5D7EC9;
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    .chat-message {
        background-color: #3D3D3D;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .source-info {
        font-size: 0
                font-size: 0
.8em;
        color: #BBBBBB;
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data(ttl=3600)
def process_document(file_content: bytes, filename: str) -> Dict[str, Any]:
    files = {"file": (filename, file_content, "application/pdf")}
    response = requests.post(f"{BACKEND_URL}/process_document", files=files)
    return response.json(), response.status_code

@st.cache_data(ttl=300)
def query_system(query: str, conversation_history: list) -> Dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}/query", json={"query": query, "conversation_history": conversation_history})
    return response.json(), response.status_code

@st.cache_data(ttl=60)
def check_system_health() -> Dict[str, Any]:
    response = requests.get(f"{BACKEND_URL}/health")
    return response.json(), response.status_code



def format_ai_response(response: str) -> str:
    # Use f-strings for better readability
    formatted = f"{response}"
    
    # Replace newlines with HTML line breaks
    formatted = formatted.replace('\n', '<br>')
    
    # Format code blocks
    formatted = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code class="\1">\2</code></pre>', formatted, flags=re.DOTALL)
    
    # Format inline code
    formatted = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', formatted)
    
    # Format bold text
    formatted = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', formatted)
    
    # Format italic text
    formatted = re.sub(r'\*([^*\n]+)\*', r'<em>\1</em>', formatted)
    
    return formatted



def main():
    # Initialize session state
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0

    # Sidebar content
    with st.sidebar:
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                file_content = uploaded_file.read()
                result, status_code = process_document(file_content, uploaded_file.name)
                if status_code == 200:
                    st.success("Document processed successfully!")
                else:
                    st.error(f"Error processing document: {result.get('error', 'Unknown error')}")
        
        st.markdown("### 📊 Statistics")
        st.metric("Queries Made", st.session_state.query_count)
        
        st.markdown("### 🔧 System")
        if st.button("Check System Health"):
            with st.spinner("Checking system health..."):
                result, status_code = check_system_health()
                if status_code == 200:
                    st.success("System is healthy!")
                else:
                    st.error("System health check failed.")
        
        show_info = st.checkbox("Show System Information")
        if show_info:
            st.info("""
            This AI Assistant uses advanced NLP techniques:
            
            - **Contextual Retrieval**: Enhanced context
            - **Hybrid Search**: Dense & sparse methods
            - **Reranking**: Improved relevance
            - **Generation**: Coherent answers via Gemini
            """)

    # Main content
    if not st.session_state.chat_started:
        st.markdown("""
        <div class="main-card">
            <h1>🧠 Contextual RAG Assistant</h1>
            <p>Experience the power of advanced retrieval and generation</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat interface
    st.header("💬 Chat with AI Assistant")

    # Display conversation history
    for turn in st.session_state.conversation_history:
        with st.chat_message("human"):
            st.markdown(f'<div class="chat-message">{turn["human"]}</div>', unsafe_allow_html=True)
        with st.chat_message("ai"):
            st.markdown(f'<div class="chat-message">{turn["ai"]}</div>', unsafe_allow_html=True)

    # Input for new question
    user_input = st.chat_input("Ask a question or type 'clear' to start a new conversation:")

    if user_input:
        if user_input.lower() == 'clear':
            st.session_state.chat_started = False
            st.session_state.conversation_history = []
            st.experimental_rerun()
        else:
            st.session_state.chat_started = True
            st.session_state.query_count += 1
            
            with st.chat_message("human"):
                st.markdown(f'<div class="chat-message">{user_input}</div>', unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                result, status_code = query_system(user_input, st.session_state.conversation_history)
            
            if status_code == 200:
                with st.chat_message("ai"):
                    # Format the AI response
                    formatted_response = format_ai_response(result["response"])
                    st.markdown(f'<div class="chat-message">{formatted_response}</div>', unsafe_allow_html=True)
                    
                    with st.expander("View sources"):
                        for i, context in enumerate(result["context"], 1):
                            st.markdown(f"""
                            <div class="source-info">
                                <strong>Source {i}:</strong> {context['source']}<br>
                                <strong>Relevance:</strong> {context['rerank_score']:.4f}<br>
                                <strong>Extract:</strong> {context['content'][:200]}...
                            </div>
                            """, unsafe_allow_html=True)

                # Update conversation history
                st.session_state.conversation_history.append({
                    "human": user_input,
                    "ai": formatted_response
                })
            else:
                st.error("Error querying the system. Please try again.")

if __name__ == "__main__":
    main()
