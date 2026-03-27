import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import using absolute paths
from src.rag_agent.vectorstore.store import VectorStoreManager
from src.rag_agent.corpus.chunker import DocumentChunker

st.set_page_config(page_title="RAG Prep", layout="wide")
st.title("🧠 Deep Learning RAG Interview Prep")

# Initialize
if "store" not in st.session_state:
    with st.spinner("Initializing..."):
        st.session_state.store = VectorStoreManager()
        st.session_state.context = None
        st.session_state.question = None

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    st.markdown("**Supported formats:** MD, TXT, PDF")
    
    files = st.file_uploader(
        "Choose files",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True
    )
    
    if files:
        chunker = DocumentChunker()
        for f in files:
            if st.button(f"Ingest {f.name}", key=f"btn_{f.name}"):
                with st.spinner(f"Ingesting {f.name}..."):
                    topic = f.name.replace('.md', '').replace('.txt', '').replace('.pdf', '')
                    
                    if f.name.lower().endswith('.pdf'):
                        chunks = chunker.chunk_pdf(f, f.name, topic)
                    elif f.name.lower().endswith('.md'):
                        content = f.read().decode('utf-8')
                        chunks = chunker.chunk_markdown(content, f.name, topic)
                    else:
                        content = f.read().decode('utf-8')
                        chunks = chunker.chunk_text(content, f.name, topic)
                    
                    if chunks:
                        result = st.session_state.store.ingest(chunks)
                        st.success(f"✅ Ingested {result['ingested']} chunks from {f.name}")
                    else:
                        st.warning(f"⚠️ No valid chunks extracted from {f.name}")
                    st.rerun()
    
    st.markdown("---")
    docs = st.session_state.store.list_documents()
    if docs:
        st.write("**Documents in Store:**")
        for d in docs:
            st.write(f"- {d}")
    else:
        st.info("No documents yet")

# Main columns
col1, col2 = st.columns(2)

with col1:
    st.header("Select Content")
    docs = st.session_state.store.list_documents()
    if docs:
        selected = st.selectbox("Choose document", docs)
        chunks = st.session_state.store.get_document_chunks(selected)
        if chunks:
            st.write(f"**{len(chunks)} chunks available**")
            for i, chunk in enumerate(chunks[:5]):
                with st.expander(f"Chunk {i+1}"):
                    preview = chunk['chunk_text'][:300]
                    if len(chunk['chunk_text']) > 300:
                        preview += "..."
                    st.write(preview)
                    if st.button(f"Use This", key=f"btn_chunk_{i}"):
                        st.session_state.context = chunk['chunk_text']
                        st.session_state.question = None
                        st.success("✅ Context loaded! Go to Practice section.")
                        st.rerun()
        else:
            st.info("No chunks found in this document")
    else:
        st.info("Upload a document first")

with col2:
    st.header("Practice")
    if st.session_state.context:
        st.success("Context ready!")
        with st.expander("View Selected Context"):
            preview = st.session_state.context[:400]
            if len(st.session_state.context) > 400:
                preview += "..."
            st.write(preview)
            if st.button("Clear Context"):
                st.session_state.context = None
                st.session_state.question = None
                st.rerun()
        
        if st.button("🎯 Generate Interview Question"):
            ctx = st.session_state.context.lower()
            
            if "backprop" in ctx or "backward" in ctx:
                q = "Explain how backpropagation works in neural networks. Include the forward pass, backward pass, and how weights are updated."
                st.session_state.expected_keywords = ["forward", "backward", "gradient", "weight", "error", "loss"]
            elif "activation" in ctx:
                q = "Compare and contrast ReLU, Sigmoid, and Tanh activation functions. When would you use each?"
                st.session_state.expected_keywords = ["relu", "sigmoid", "tanh", "nonlinear", "vanishing", "gradient"]
            elif "gradient" in ctx or "descent" in ctx:
                q = "What is gradient descent? Explain how the learning rate affects training and what happens if it's too high or too low."
                st.session_state.expected_keywords = ["gradient", "descent", "learning rate", "optimization", "converge", "overshoot"]
            elif "neural" in ctx or "network" in ctx:
                q = "Describe the basic architecture of a neural network. What are the key components and how do they work together?"
                st.session_state.expected_keywords = ["layer", "neuron", "weight", "activation", "input", "output", "hidden"]
            else:
                q = "Summarize the key concepts from this material in your own words."
                st.session_state.expected_keywords = ["concept", "explain", "describe", "key"]
            
            st.session_state.question = q
            st.info(f"**Question:** {q}")
        
        if st.session_state.question:
            answer = st.text_area("Your answer:", height=150, key="answer_area")
            
            if st.button("Submit Answer"):
                if answer.strip():
                    answer_lower = answer.lower()
                    expected_keywords = st.session_state.get("expected_keywords", [])
                    
                    matched_keywords = []
                    for keyword in expected_keywords:
                        if keyword in answer_lower:
                            matched_keywords.append(keyword)
                    
                    if expected_keywords:
                        score = int((len(matched_keywords) / len(expected_keywords)) * 10)
                    else:
                        word_count = len(answer.split())
                        if word_count > 100:
                            score = 9
                        elif word_count > 50:
                            score = 7
                        elif word_count > 20:
                            score = 5
                        else:
                            score = 3
                    
                    score = max(1, min(10, score))
                    
                    if score >= 9:
                        feedback = "🎉 Excellent! Comprehensive answer covering all key concepts."
                    elif score >= 7:
                        feedback = "👍 Good answer! You covered most key points. Review the material for more details."
                    elif score >= 5:
                        feedback = "📚 Decent effort, but missing some important concepts."
                    else:
                        feedback = "📖 Needs improvement. Review the material and try to include key concepts."
                    
                    st.success(f"**Score: {score}/10**")
                    st.info(feedback)
                    
                    if matched_keywords and score < 7:
                        st.write("**Key concepts to include:**")
                        missing = [k for k in expected_keywords if k not in matched_keywords]
                        for m in missing[:5]:
                            st.write(f"- {m}")
                    
                    st.session_state.question = None
                    st.session_state.expected_keywords = None
                    st.rerun()
                else:
                    st.warning("Please provide an answer.")
    else:
        st.info("Select a chunk from the left to start practicing")
