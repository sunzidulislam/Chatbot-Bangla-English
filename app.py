import streamlit as st
import pandas as pd
import base64
import sys
from datetime import datetime

import fitz  # pip install pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Ensure stdout utf-8 encoding for Bangla
sys.stdout.reconfigure(encoding='utf-8')

def get_pdf_text(pdf_docs):
    """Extract text from PDF docs using PyMuPDF (fitz)."""
    text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    cleaned = page_text.replace('\xa0', ' ').strip()
                    text += cleaned + "\n"
    return text

def preprocess_text(text):
    """Basic cleaning to normalize text for better chunking."""
    # Remove excessive spaces and normalize newlines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)  # max two newlines
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def get_text_chunks(text, model_name):
    """Split text into chunks for vector store."""
    if model_name == "Google AI":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "।", "!", "?"]
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    else:
        raise ValueError("Unsupported model")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
        আপনি একজন বুদ্ধিমান ও বিশ্বস্ত সহকারী। নিচের তথ্য থেকে প্রশ্নের সঠিক উত্তর বাংলায় বা ইংরেজিতে দিন।
        যদি উত্তর পাওয়া না যায়, তাহলে বলুন: "এই প্রশ্নের উত্তর প্রসঙ্গের মধ্যে নেই"। ভুল তথ্য দেবেন না।

        প্রাসঙ্গিক তথ্য:
        {context}

        প্রশ্ন:
        {question}

        উত্তর:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    else:
        raise ValueError("Unsupported model")

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("PDF আপলোড করুন এবং API key দিন।")
        return
    
    raw_text = get_pdf_text(pdf_docs)
    raw_text = preprocess_text(raw_text)
    text_chunks = get_text_chunks(raw_text, model_name)
    
    # For debug: Show number of chunks
    st.sidebar.write(f"🔢 মোট চাংক সংখ্যা: {len(text_chunks)}")
    
    vector_store = get_vector_store(text_chunks, model_name, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=10)

    # Debug: Show matched chunks for user
    for i, doc in enumerate(docs):
        st.sidebar.text_area(f"প্রাসঙ্গিক অংশ {i+1}", doc.page_content[:1500], height=150)

    chain = get_conversational_chain(model_name, vectorstore=new_db, api_key=api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    user_question_output = user_question
    response_output = response['output_text']

    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    # Display latest QA
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
                white-space: pre-wrap;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png" alt="User"/>
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" alt="Bot" />
            </div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show previous conversation history
    for question, answer, m_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png" alt="User"/>
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" alt="Bot"/>
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # CSV Download of history
    if len(conversation_history) > 0:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("CSV ডাউনলোড করতে বাম পাশে বাটনে ক্লিক করুন।")

    st.snow()

def main():
    st.set_page_config(page_title="বাংলা PDF প্রশ্নোত্তর", page_icon="📚")
    st.header("বাংলা PDF প্রশ্নোত্তর বট 📚")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("মডেল নির্বাচন করুন:", ("Google AI",))

    api_key = None
    if model_name == "Google AI":
        api_key = st.sidebar.text_input("আপনার Google API Key লিখুন:", type="password")
        st.sidebar.markdown("API key পেতে [এখানে যান](https://ai.google.dev/)।")
        if not api_key:
            st.sidebar.warning("চালানোর জন্য Google API Key দিন।")
            return

    with st.sidebar:
        st.title("নিয়ন্ত্রণ মেনু:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("রিসেট")
        clear_button = col1.button("পুনরায় চালান")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
            api_key = None
            pdf_docs = None
        elif clear_button:
            if 'user_question' in st.session_state:
                st.warning("আগের প্রশ্ন বাতিল হবে।")
                st.session_state.user_question = ""
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader("আপনার বাংলা PDF আপলোড করুন", accept_multiple_files=True)
        if st.button("সাবমিট ও প্রসেস করুন"):
            if pdf_docs:
                with st.spinner("প্রসেসিং হচ্ছে..."):
                    st.success("সম্পন্ন হয়েছে ✅")
            else:
                st.warning("দয়া করে PDF আপলোড করুন।")

    user_question = st.text_input("PDF থেকে প্রশ্ন করুন (বাংলায়/English)")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""

if __name__ == "__main__":
    main()
