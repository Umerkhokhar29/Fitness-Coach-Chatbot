import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FitnessCoachRAG:
    def __init__(self, csv_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG Fitness Coach"""
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            st.error("Please set OPENAI_API_KEY in your environment variables or Streamlit secrets")
            st.stop()
        
        self.client = OpenAI(api_key=api_key)
        
        # Load and process data
        self._load_data(csv_path)
        logger.info("Fitness Coach RAG initialized successfully!")
    
    def _load_data(self, csv_path: str):
        """Load CSV data and create embeddings"""
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                st.error(f"CSV file not found: {csv_path}")
                st.stop()
            
            df = pd.read_csv(csv_path)
            
            # Handle missing values and clean data
            if 'content' not in df.columns:
                # Try to find the right column
                possible_columns = ['content', 'text', 'document', 'data', 'description']
                content_col = None
                for col in possible_columns:
                    if col in df.columns:
                        content_col = col
                        break
                
                if content_col is None:
                    st.error(f"Could not find content column. Available columns: {list(df.columns)}")
                    st.stop()
                
                df = df.rename(columns={content_col: 'content'})
            
            df = df.dropna(subset=['content'])
            df['content'] = df['content'].astype(str).str.strip()
            df = df[df['content'] != '']
            
            self.documents = df['content'].tolist()
            logger.info(f"Loaded {len(self.documents)} documents from {csv_path}")
            
            # Create embeddings
            self._create_embeddings()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Error loading data: {e}")
            st.stop()
    
    def _create_embeddings(self):
        """Create embeddings for all documents"""
        try:
            # Check if embeddings cache exists
            cache_file = "fitness_embeddings.pkl"
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        if len(cached_data['documents']) == len(self.documents):
                            self.embeddings = cached_data['embeddings']
                            st.success("✅ Loaded cached embeddings")
                            return
                except:
                    pass  # If cache loading fails, create new embeddings
            
            # Create new embeddings
            batch_size = 32
            all_embeddings = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(self.documents), batch_size):
                batch_docs = self.documents[i:i+batch_size]
                batch_embeddings = self.embedder.encode(batch_docs, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min(i + batch_size, len(self.documents)) / len(self.documents)
                progress_bar.progress(progress)
                status_text.text(f"Creating embeddings: {min(i+batch_size, len(self.documents))}/{len(self.documents)}")
            
            self.embeddings = np.array(all_embeddings)
            
            # Cache embeddings for faster future loads
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'embeddings': self.embeddings
                    }, f)
            except:
                pass  # If caching fails, it's not critical
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            logger.error(f"Error creating embeddings: {e}")
            st.stop()
    
    def get_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k similar documents using cosine similarity"""
        try:
            # Create query embedding
            query_embedding = self.embedder.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Filter by similarity threshold (you can adjust this)
            threshold = 0.3  # Minimum similarity score
            filtered_docs = []
            
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    filtered_docs.append(self.documents[idx])
            
            # Return at least 2 documents even if below threshold
            if not filtered_docs:
                filtered_docs = [self.documents[idx] for idx in top_indices[:2]]
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return self.documents[:2] if self.documents else []
    
    def ask_coach(self, query: str, conversation_history: List = None) -> str:
        """Get response from the fitness coach"""
        try:
            # Get relevant context
            context_chunks = self.get_context(query)
            context = "\n---\n".join(context_chunks)
            
            # Build messages
            messages = [
                {
                    "role": "system", 
                    "content": """You are an expert fitness coach with years of experience helping people achieve their health and fitness goals. 

Guidelines:
- Provide personalized, actionable advice based on the knowledge context
- Always prioritize safety and proper form
- Ask follow-up questions when you need more information
- Be encouraging and motivational
- If the context doesn't contain relevant information, use your general fitness knowledge but mention this
- Keep responses concise but comprehensive
- Include specific recommendations when possible"""
                }
            ]
            
            # Add conversation history (last 4 exchanges)
            if conversation_history:
                recent_history = conversation_history[-8:]  # Last 4 exchanges (user + assistant)
                for msg in recent_history:
                    messages.append(msg)
            
            # Add current query with context
            user_message = f"""Context from fitness knowledge base:
{context}

User Question: {query}"""
            
            messages.append({"role": "user", "content": user_message})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )

            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "I'm sorry, I encountered an error. Please try asking your question again."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "coach" not in st.session_state:
    st.session_state.coach = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.title("🏋️ Fitness Coach Settings")
    
    # CSV file upload
    uploaded_file = st.file_uploader(
        "Upload your fitness data CSV",
        type=['csv'],
        help="Upload a CSV file with fitness knowledge data"
    )
    
    # CSV path input as fallback
    csv_path = st.text_input(
        "Or enter CSV file path:",
        value="fitness_data_fixed2.csv",
        help="Path to your fitness data CSV file"
    )
    
    # Model selection
    model_option = st.selectbox(
        "Choose embedding model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        help="Different models offer trade-offs between speed and accuracy"
    )
    
    # Initialize button
    if st.button("Initialize Coach", type="primary"):
        with st.spinner("Initializing your AI Fitness Coach..."):
            try:
                file_path = csv_path
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with open("temp_fitness_data.csv", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_path = "temp_fitness_data.csv"
                
                st.session_state.coach = FitnessCoachRAG(file_path, model_option)
                st.session_state.initialized = True
                st.success("✅ Coach initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Error initializing coach: {e}")
    
    st.divider()
    
    # Quick actions
    st.subheader("🎯 Quick Actions")
    
    if st.button("💡 Get General Tips"):
        if st.session_state.initialized:
            st.session_state.messages.append({
                "role": "user", 
                "content": "What are some general fitness tips for beginners?"
            })
            st.rerun()
    
    if st.button("🏃 Cardio Advice"):
        if st.session_state.initialized:
            st.session_state.messages.append({
                "role": "user", 
                "content": "What's the best cardio routine for weight loss?"
            })
            st.rerun()
    
    if st.button("💪 Strength Training"):
        if st.session_state.initialized:
            st.session_state.messages.append({
                "role": "user", 
                "content": "How should I start strength training as a beginner?"
            })
            st.rerun()
    
    if st.button("🥗 Nutrition Tips"):
        if st.session_state.initialized:
            st.session_state.messages.append({
                "role": "user", 
                "content": "What should I eat to support my fitness goals?"
            })
            st.rerun()
    
    st.divider()
    
    # Clear chat
    if st.session_state.initialized and st.session_state.coach:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Stats and info
    if st.session_state.messages:
        st.subheader("📊 Chat Stats")
        st.metric("Messages", len(st.session_state.messages))
    
    if st.session_state.initialized and st.session_state.coach:
        st.subheader("📋 Data Info")
        st.metric("Documents Loaded", len(st.session_state.coach.documents))

# Main interface
st.title("💪 AI Fitness Coach")
st.markdown("Your personal fitness companion powered by AI and expert knowledge!")

if not st.session_state.initialized:
    st.info("👈 Please initialize your coach using the sidebar first!")
    
    # Show some example questions
    st.subheader("🤔 Example Questions You Can Ask:")
    examples = [
        "What's the best workout routine for building muscle?",
        "How can I lose weight effectively?",
        "What should I eat before and after workouts?",
        "How do I improve my running endurance?",
        "What are some good exercises for back pain?",
        "How often should I work out each week?"
    ]
    
    col1, col2 = st.columns(2)
    for i, example in enumerate(examples):
        if i % 2 == 0:
            col1.markdown(f"• {example}")
        else:
            col2.markdown(f"• {example}")

else:
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your fitness question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get coach response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert messages to format expected by coach
                coach_history = []
                for msg in st.session_state.messages[:-1]:  # Exclude the current message
                    coach_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                response = st.session_state.coach.ask_coach(prompt, coach_history)
                st.markdown(response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🏋️ Built with Streamlit • Powered by OpenAI & SentenceTransformers • Stay Strong! 💪
    </div>
    """,
    unsafe_allow_html=True
)