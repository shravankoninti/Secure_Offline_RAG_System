import streamlit as st
import pandas as pd
import torch
from typing import List
from pathlib import Path
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader, PyPDFLoader, TextLoader, DirectoryLoader
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
import time
from fpdf import FPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add page configuration
st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_embedding_model():
    time.sleep(2)  # Simulate loading time
    return "Embedding Model"

def load_language_model():
    time.sleep(2)  # Simulate loading time
    return "Language Model"

class MultilingualRAGSystem:
    def __init__(self):
        self.conversation = []
        try:
            # Create placeholders for loading messages
            loading_placeholder = st.empty()
            loading_placeholder.info("Initializing RAG System...")
            self.embeddings = None
            self.vectorstore = None
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            # Create placeholders for loading messages
            loading_placeholder.empty()  # Clear the loading message
            st.success("RAG System initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing RAG System: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            raise

    # Function to create a PDF from the chat history
    def create_pdf(self, chat_history):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Use built-in fonts instead of trying to load external fonts
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        def clean_text(text):
            """Clean and encode text to be PDF-safe"""
            # Replace problematic characters
            text = text.encode('latin-1', 'replace').decode('latin-1')
            return text

        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt='Chat Conversation History', ln=True, align='C')
        pdf.ln(10)

        for message in chat_history:
            # Format the role
            pdf.set_font("Arial", 'B', 12)
            role = message["role"].capitalize() + ": "
            pdf.cell(0, 10, txt=clean_text(role), ln=True)
            
            # Format the content
            pdf.set_font("Arial", '', 12)
            content = clean_text(message["content"])
            pdf.multi_cell(0, 10, txt=content)
            pdf.ln(5)

        try:
            return pdf.output(dest='S').encode('latin-1')
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            # Create a simple error PDF if generation fails
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Error: Could not generate PDF with special characters.", ln=True)
            return pdf.output(dest='S').encode('latin-1')

    def initialize_models(self):
        """Initialize embedding and language models"""
        try:
            # Create placeholders for loading messages
            loading_placeholder = st.empty()
            # Show loading messages
            loading_placeholder.info("Loading embedding model...")
            embedding_model = load_embedding_model()
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            # Clear loading message and show success
            loading_placeholder.empty()  # Clear the loading message
            st.success("Embedding model loaded!")

            # Show loading message for language model
            loading_placeholder.info("Loading language model...")
            language_model = load_language_model()
            model_name = "Qwen/Qwen2.5-14B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            # Clear loading message and show success
            loading_placeholder.empty()  # Clear the loading message
            st.success("Language model loaded successfully!")

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            logger.error(f"Model loading error: {str(e)}")
            raise

    def load_documents(self, uploaded_files) -> List:
        """Load and process documents from various sources"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    documents.extend([Document(page_content=str(row), metadata={"source": uploaded_file.name, "row": idx}) for idx, row in df.iterrows()])
                elif uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif uploaded_file.name.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
                
                os.unlink(file_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                logger.error(f"Document loading error: {str(e)}")
                continue
        
        return documents

    def process_documents(self, documents: List[Document]) -> None:
        """Process documents and create vector store"""
        try:
            if not documents:
                st.warning("No documents to process!")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            st.success(f"Processed {len(splits)} document chunks!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            logger.error(f"Document processing error: {str(e)}")
            raise


    def post_process_response(self, response):
    # Check if the response ends with a complete sentence
        if not response.endswith(('.', '!', '?')):
            response += '.'  # Append a period if it doesn't end with a punctuation
        return response
    
    def get_response(self, query: str, language: str = "en") -> str:
        try:
            def is_greeting(text: str, lang: str) -> bool:
                from difflib import get_close_matches
                
                greetings = {
                    "en": [
                        "hello", "hi", "hey", "good morning", "good afternoon", 
                        "good evening", "how are you", "what's up", "whats up",
                        "how're you", "how are you doing", "greetings", "morning",
                        "evening", "afternoon", "yo", "sup", "hiya", "howdy"
                    ],
                    "fr": [
                        "bonjour", "salut", "coucou", "bonsoir", "comment allez-vous",
                        "comment vas-tu", "ça va", "ca va", "comment ça va",
                        "bjr", "slt", "bsr", "cc"
                    ]
                }
                
                # Get greetings for specified language
                valid_greetings = greetings.get(lang.lower(), greetings["en"])
                
                # Clean and normalize input text
                text = text.lower().strip().rstrip('?!.,')
                words = text.split()
                
                # Check if the input contains only greeting words
                # If the input has question words or is longer than 3 words, it's probably not just a greeting
                question_words = ["what", "why", "how", "when", "where", "which", "who", "whose"]
                if any(word in text for word in question_words) or len(words) > 3:
                    return False
                
                # Check for greeting matches
                for word in words:
                    matches = get_close_matches(word, valid_greetings, n=1, cutoff=0.7)
                    if matches:
                        return True
                
                return False

            def get_greeting_response(lang: str) -> str:
                import random
                responses = {
                    "en": [
                        "Hello! How can I assist you today?",
                        "Hi there! What can I help you with?",
                        "Greetings! How may I be of assistance?",
                        "Hello! I'm here to help. What do you need?",
                        "Hi! Please let me know how I can help you today."
                    ],
                    "fr": [
                        "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                        "Salut ! Que puis-je faire pour vous ?",
                        "Bonjour ! En quoi puis-je vous être utile ?",
                        "Salut ! Je suis là pour vous aider. Que puis-je faire pour vous ?",
                        "Bonjour ! Dites-moi comment je peux vous aider."
                    ]
                }
                return random.choice(responses.get(lang.lower(), responses["en"]))

            # Always check for vectorstore first for non-greeting queries
            if not is_greeting(query, language):
                if not self.vectorstore:
                    return "Please load documents first."

            # Handle greetings
            if is_greeting(query, language):
                return get_greeting_response(language)
                
            # Process document-based queries
            relevant_docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            if not context:
                return "I couldn't find any relevant information. Please ask something else!"

            templates = {
                "en": f"Based on the provided context, please provide a detailed and complete answer to the following question: {query}\nContext: {context}\nEnsure that your response is coherent and ends with a complete thought.\nAnswer: ",
                "fr": f"En vous basant sur le contexte fourni, veuillez donner une réponse détaillée et complète à la question suivante : {query}\nContexte : {context}\nAssurez-vous que votre réponse est cohérente et se termine par une pensée complète.\nRéponse : "
            }
            
            response = self.pipeline(templates.get(language, templates["en"]))[0]['generated_text'].strip()
            final_answer = response.split("Answer:")[-1].strip()
            final_answer = self.post_process_response(final_answer)
            return final_answer if final_answer else "I couldn't find a specific answer to your question."

        except Exception as e:
            return f"An error occurred: {str(e)}"
      



def main():
    try:
        st.title("🤖 Multilingual - Secure Offline RAG Chat Interface System")

        # Sidebar for user details
        st.sidebar.header("Author Information")
        st.sidebar.markdown('[![Shravan-Koninti]'
        '(https://img.shields.io/badge/Author-Shravan%20Koninti-brightgreen)]'
        '(https://www.linkedin.com/in/shravankoninti/)')
        st.sidebar.markdown("[Email](mailto:shravankumar224@gmail.com)")

       
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.rag_system = MultilingualRAGSystem()
                st.session_state.rag_system.initialize_models()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.sidebar:
            st.header("📁 Document Upload")
            uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['csv', 'pdf', 'txt'])
            language = st.selectbox("🌍 Response Language", options=["en", "fr"], format_func=lambda x: "English" if x == "en" else "French")
            
            if uploaded_files:
                if st.button("📥 Process Documents"):
                    with st.spinner("Processing documents..."):
                        documents = st.session_state.rag_system.load_documents(uploaded_files)
                        st.session_state.rag_system.process_documents(documents)
        
        st.header("💬 Chat Interface")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if query := st.chat_input("Ask a question about your documents"):
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_system.get_response(query, language)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Add the Download Chat as PDF button
        if st.button("Download Chat as PDF"):
            with st.spinner("Generating PDF..."):
                try:
                    if st.session_state.chat_history:
                        pdf_data = st.session_state.rag_system.create_pdf(st.session_state.chat_history)
                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name='chat_conversation.pdf',
                            mime='application/pdf',
                        )
                    else:
                        st.warning("No conversation to save!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    logger.error(f"PDF generation error: {str(e)}")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()