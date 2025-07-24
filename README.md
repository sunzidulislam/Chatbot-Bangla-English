# 🤖 Chatbot-Bangla-English

📚 **Bangla-English PDF-Based RAG QA System**

A bilingual **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from Bangla and English PDF documents. It combines **Google Gemini**, **LangChain**, and **FAISS** for accurate and context-aware responses.

---

## 🚀 Features

* 🌐 Multilingual support (Bangla 🇧🇩 & English 🇺🇸)
* 📄 PDF-based document retrieval
* 🤖 Google Gemini-powered answer generation
* ✅ RAG Evaluation (Groundedness & Relevance)
* 🧠 Dual memory: Long-term (FAISS) + Short-term (Chat history)

---

## 🔑 Prerequisites

* Python 3.9+
* Git installed
* Google Gemini API key → [Get API Key](https://ai.google.dev/)

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bangla-rag-chatbot.git
cd bangla-rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Or install manually</summary>

```bash
pip install streamlit pymupdf langchain faiss-cpu pandas scikit-learn langchain-google-genai
```

</details>

### 4. Run the App

```bash
python -m streamlit run app.py
```

---

## 🛠️ How to Use

1. Enter your **Google API Key**
2. Upload a **Bangla or English PDF** (e.g., *HSC26 Bangla 1st Paper*)
3. Ask questions in **Bangla or English**
4. Get:

   * Relevant answer from Gemini
   * Evaluation scores (Groundedness & Relevance)

---

## 🧪 Evaluation (Optional)

This app includes **basic evaluation** of RAG answers using cosine similarity or manually labeled relevance metrics. Modify `evaluate.py` or add your own evaluation logic.

---

## 📁 File Structure

```
├── app.py                # Main Streamlit app
├── rag_pipeline.py       # RAG logic (embedding, retrieval, generation)
├── evaluate.py           # Optional evaluation functions
├── requirements.txt      # Project dependencies
├── data/                 # PDF files or sample data
└── README.md             # Project documentation
```

---

## 📦 requirements.txt

```txt
streamlit
pymupdf
langchain
faiss-cpu
pandas
scikit-learn
langchain-google-genai
```

---

## 📹 Demo (Optional)

> 📽️ Add a short demo video link here showing how to use the chatbot.

---

## 👨‍💻 Author

**Md. Sunzidul Islam**
*AI Researcher & Developer*
[GitHub Profile](https://github.com/your-username)

---

## 📬 Contact

For issues, open a GitHub issue or reach out via email.

---
