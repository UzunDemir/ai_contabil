import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer

# Инициализация
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
st.set_page_config(layout="wide", page_title="Бухгалтерский AI-советник")

class DocumentChunk:
    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.doc_texts = []
    
    def split_text(self, text, max_tokens=1500):
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            tokens = tokenizer.tokenize(para)
            if len(tokenizer.tokenize(current_chunk + para)) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    chunks.append(para)
                    current_chunk = ""
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

# Инициализация сессии
if 'kb' not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Интерфейс
st.title("🧮 Бухгалтерский AI-советник Молдовы")
st.write("Загрузите PDF с законодательными актами для анализа")

# Загрузка файлов
uploaded_files = st.file_uploader(
    "Выберите файлы законов (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files and 'kb' in st.session_state:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in [chunk.doc_name for chunk in st.session_state.kb.chunks]:
            continue
            
        try:
            with BytesIO(uploaded_file.getvalue()) as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                chunks = st.session_state.kb.split_text(text)
                for chunk in chunks:
                    st.session_state.kb.chunks.append(DocumentChunk(
                        text=chunk,
                        doc_name=uploaded_file.name,
                        page_num=0
                    ))
                    st.session_state.kb.doc_texts.append(chunk)
                
            st.success(f"✅ Успешно загружен: {uploaded_file.name}")
            st.session_state.kb.tfidf_matrix = st.session_state.kb.vectorizer.fit_transform(
                st.session_state.kb.doc_texts
            )
        except Exception as e:
            st.error(f"❌ Ошибка загрузки {uploaded_file.name}: {str(e)}")

# Чат
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ваш вопрос по бухучету/налогам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if not st.session_state.kb.chunks:
        response = "⚠️ Сначала загрузите законодательные акты"
    else:
        # Поиск релевантных фрагментов
        query_vec = st.session_state.kb.vectorizer.transform([prompt])
        similarities = cosine_similarity(query_vec, st.session_state.kb.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-3:][::-1]
        
        relevant_chunks = [
            (st.session_state.kb.chunks[i].text, 
             st.session_state.kb.chunks[i].doc_name, 
             st.session_state.kb.chunks[i].page_num)
            for i in top_indices if similarities[0][i] > 0.1
        ]
        
        if not relevant_chunks:
            response = "❌ Ответ не найден в загруженных документах"
        else:
            context = "\n\n".join([f"📄 {doc} (стр. {page}):\n{text}" 
                                 for text, doc, page in relevant_chunks])
            
            # Имитация ответа (в реальном коде здесь будет вызов API)
            response = f"""На основании законодательства:

1. Основное положение...
2. Дополнительные требования...

🔍 Источники:
- {relevant_chunks[0][1]} (стр. {relevant_chunks[0][2]})
- {relevant_chunks[1][1]} (стр. {relevant_chunks[1][2]})"""
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

if st.button("Очистить историю"):
    st.session_state.messages = []
    st.rerun()
