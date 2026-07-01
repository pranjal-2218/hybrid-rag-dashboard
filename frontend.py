import streamlit as st
import requests
import time

# Page Configuration for Premium Aesthetic
st.set_page_config(
    page_title="Advanced Hybrid RAG Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for Modern Dark Mode & Custom Elements
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Typography Override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling with vibrant gradient */
    .header-container {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C084FC 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.25);
    }
    
    .header-container h1 {
        font-weight: 800;
        margin: 0;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    
    .header-container p {
        font-weight: 300;
        margin: 0.75rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Custom Card Design for Retrieve Results */
    .result-card {
        background-color: #1E1E2E;
        border: 1px solid #2D2D3E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(124, 58, 237, 0.15);
        border-color: #7C3AED;
    }
    
    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #2D2D3E;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .badge-doc {
        background: linear-gradient(90deg, #3730A3 0%, #1E1B4B 100%);
        color: #E0E7FF;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #4F46E5;
    }
    
    .badge-score {
        background: linear-gradient(90deg, #065F46 0%, #022C22 100%);
        color: #D1FAE5;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #10B981;
    }
    
    .chunk-content {
        color: #F1F5F9;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .metric-bubble {
        background-color: #181825;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        text-align: center;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API Engine endpoints configurations
import os

RENDER_BACKEND_URL = "https://hybrid-rag-backend.onrender.com"

if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER") or os.path.exists('/.dockerenv'):
    BACKEND_BASE_URL = "http://backend:8000"
elif os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT") == "cloud":
    BACKEND_BASE_URL = RENDER_BACKEND_URL
else:
    BACKEND_BASE_URL = "http://localhost:8000"

INGEST_URL = f"{BACKEND_BASE_URL}/ingest"
QUERY_URL = f"{BACKEND_BASE_URL}/query"

# Title Header
st.markdown(
    """
    <div class="header-container">
        <h1>⚡ Advanced Hybrid RAG Engine</h1>
        <p>Production-Grade Lexical + Dense Semantic Retrieval Fused with Reciprocal Rank Fusion (RRF)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration for Ingestion Console
st.sidebar.title("📥 Ingestion Console")
st.sidebar.markdown("Use this panel to chunk and index custom legal, financial, or multilingual text documents.")

# --- HYBRID INGESTION FORM BLOCK ---
with st.sidebar.form("ingestion_form", clear_on_submit=False):
    st.markdown("#### 📄 Option 1: Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text or PDF files", 
        type=["txt", "pdf"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.markdown("#### ✍️ Option 2: Paste Raw Text")
    raw_doc_id = st.text_input("Raw Document ID", placeholder="e.g., custom_snippet_1")
    raw_text_content = st.text_area("Raw Text Content", height=150, placeholder="Paste your raw text content here...")
    
    submit_ingest = st.form_submit_button("Index Content")

if submit_ingest:
    has_files = bool(uploaded_files)
    has_text = bool(raw_text_content.strip())
    
    if not has_files and not has_text:
        st.sidebar.error("⚠️ Please provide at least one source (upload a file or paste raw text)!")
    elif has_text and not raw_doc_id.strip():
        st.sidebar.error("⚠️ Please provide a Document ID for your pasted raw text!")
    else:
        total_chunks_indexed = 0
        
        # 1. Process files if present
        if has_files:
            for uploaded_file in uploaded_files:
                derived_doc_id = uploaded_file.name
                with st.spinner(f"Processing file: {derived_doc_id}..."):
                    try:
                        extracted_text = ""
                        if uploaded_file.type == "text/plain":
                            extracted_text = uploaded_file.read().decode("utf-8")
                        elif uploaded_file.type == "application/pdf":
                            import pypdf
                            pdf_reader = pypdf.PdfReader(uploaded_file)
                            extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                        
                        if not extracted_text.strip():
                            st.sidebar.warning(f"⚠️ {derived_doc_id} was empty. Skipping.")
                            continue

                        response = requests.post(
                            INGEST_URL,
                            json={"document_id": derived_doc_id, "text": extracted_text},
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            total_chunks_indexed += response.json()['num_chunks']
                        else:
                            st.sidebar.error(f"❌ Ingestion failed for {derived_doc_id}")
                            
                    except Exception as e:
                        st.sidebar.error(f"⚠️ System error on {derived_doc_id}: {str(e)}")

        # 2. Process pasted raw text if present
        if has_text:
            with st.spinner(f"Processing raw text: {raw_doc_id}..."):
                try:
                    response = requests.post(
                        INGEST_URL,
                        json={"document_id": raw_doc_id.strip(), "text": raw_text_content.strip()},
                        timeout=90
                    )
                    if response.status_code == 200:
                        total_chunks_indexed += response.json()['num_chunks']
                    else:
                        st.sidebar.error(f"❌ Ingestion failed for raw text: {raw_doc_id}")
                except Exception as e:
                    st.sidebar.error(f"⚠️ System error on raw text: {str(e)}")
        
        # Final confirmation
        if total_chunks_indexed > 0:
            st.sidebar.success(f"✅ Context database updated successfully!")
            st.sidebar.markdown(
                f"""
                <div class="metric-bubble">
                    Total New Chunks Created: <span style="color:#C084FC">{total_chunks_indexed}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
# --- END OF HYBRID INGESTION FORM BLOCK ---

# Main Interface: Query & Search
st.subheader("🔍 Hybrid Retrieval Query Sandbox")
st.markdown("Run combined semantic (vector-based) and lexical (BM25) searches over the indexed corpus.")

# Search UI inputs
col1, col2 = st.columns([4, 1])
with col1:
    query_str = st.text_input("Query String", placeholder="Search for terms, concepts, or semantic ideas...")
with col2:
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5, step=1)

run_search = st.button("Execute Hybrid Search", use_container_width=True)

# Process search query
if run_search or query_str:
    if not query_str.strip():
        st.warning("Please enter a query string.")
    else:
        with st.spinner("Executing RRF search (Lexical + Dense)..."):
            start_search_time = time.perf_counter()
            try:
                response = requests.post(
                    QUERY_URL,
                    json={"query": query_str.strip(), "top_k": top_k},
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    elapsed = time.perf_counter() - start_search_time
                    
                    if not results:
                        st.info("ℹ️ No matching chunks found. Try indexing a document first or adjusting your query.")
                    else:
                        st.markdown(f"**Retrieved {len(results)} chunks in {elapsed:.4f} seconds:**")
                        
                        for idx, item in enumerate(results, start=1):
                            # HTML injection for custom styled cards
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="card-top-row">
                                        <div>
                                            <span style="font-weight: 800; color: #7C3AED; margin-right: 0.5rem;">#{idx}</span>
                                            <span class="badge-doc">Document: {item['document_id']}</span>
                                        </div>
                                        <div>
                                            <span class="badge-score">RRF Score: {item['score']:.5f}</span>
                                        </div>
                                    </div>
                                    <p class="chunk-content">{item['text']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    error_detail = response.json().get("detail", "Unknown error occurred.")
                    st.error(f"❌ Query Failed: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Connection Error: Is the FastAPI backend running on port 8000?")
            except Exception as e:
                st.error(f"⚠️ Search Query Failed: {str(e)}")