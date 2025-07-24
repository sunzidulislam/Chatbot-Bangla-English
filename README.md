# Chatbot-Bangla-English
Here is a complete **Setup Guide** in `README.md` format for your Bangla-English PDF-based RAG Question Answering project with evaluation support:

---

```markdown
# 📚 Bangla-English PDF-Based RAG QA System

This project is a bilingual **Retrieval-Augmented Generation (RAG)** system that answers questions from Bangla and English PDF documents using Google Gemini, LangChain, and FAISS.

It supports:
- 📖 Multilingual queries (Bangla + English)
- 🔍 PDF-based document retrieval
- 🤖 Gemini-powered answer generation
- 📈 RAG Evaluation (Groundedness & Relevance)
- 🧠 Memory: Long-term (Vector DB) + Short-term (chat history)

---

## 🚀 Demo Screenshot

![Demo Screenshot](demo.png)

---

## 🧰 Tech Stack

| Tool            | Purpose                            |
|-----------------|-------------------------------------|
| Streamlit       | Frontend UI                        |
| LangChain       | RAG pipeline management             |
| Google Gemini   | LLM-based response generation       |
| FAISS           | Vector similarity search            |
| PyMuPDF         | PDF text extraction (`fitz`)       |
| scikit-learn    | Cosine similarity (RAG evaluation)  |

---

## 🗂️ Folder Structure

```

````
---

## 🔑 Prerequisites

- Python 3.9+
- Google Gemini API key: [Get your API key](https://ai.google.dev/)
- Git installed

---

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/bangla-rag-chatbot.git
cd bangla-rag-chatbot
````

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pymupdf langchain faiss-cpu pandas scikit-learn langchain-google-genai
```

4. **Run the Streamlit app**

```bash
 python -m streamlit run app.py
```

5. **Use the app**

   * Enter your **Google API Key**
   * Upload a Bangla/English PDF (e.g., `HSC26 Bangla 1st Paper`)
   * Ask questions in Bangla or English
   * View model answer + RAG evaluation metrics

---

## 📈 RAG Evaluation Included

| Metric       | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| Relevance    | Cosine similarity between question and retrieved chunks            |
| Groundedness | Cosine similarity between model's answer and the retrieved context |
| 🔔 Alert     | Shown when similarity scores are low (e.g., hallucination risk)    |

---

## 📁 Sample PDF

> Upload your own **Bangla 1st Paper PDF** during runtime via the UI.
> Recommended: Use clean, OCR-friendly PDFs.

---

## 🧪 Example Query

**Question:**

> "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"

**Expected Answer:**

> "15 বছর"

**RAG Evaluation:**

* Relevance Score: 0.92
* Groundedness Score: 0.88

---

## 🧪 Sample Queries (Bangla & English)

| Query                                   | Type    |
| --------------------------------------- | ------- |
| "কল্যাণীর লেখাপড়া কেন বন্ধ হয়ে যায়?" | Bangla  |
| "Why did Kalyani stop going to school?" | English |
| "সুশীল কেমন ব্যক্তি ছিলেন?"             | Bangla  |
| "What type of person was Sushil?"       | English |

---

## 📤 Export Features

* All Q\&A and RAG scores can be downloaded as CSV
* Button available in Streamlit sidebar

---

## 📦 Requirements

> `requirements.txt`

```text
streamlit
pymupdf
langchain
faiss-cpu
pandas
scikit-learn
langchain-google-genai
```

---

## 📽️ Demo Video (Optional)

> Add a screen recording of the app usage and upload it here.

---

## 👨‍💻 Author

Md. Sunzidul Islam
*AI Researcher & Developer*
[GitHub Profile](https://github.com/your-username)

---

## 📬 Contact

If you face any issues, open a GitHub issue or contact me directly via email.

```

---

Let me know if you'd like this exported as `README.md` or want a GitHub-ready version with badges and demo link placeholders.
```
