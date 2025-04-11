import os
import pickle
import hashlib
import streamlit as st
import requests
import numpy as np
import tempfile  # Добавлен отсутствующий импорт
from PyPDF2 import PdfReader
from datetime import datetime
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Конфигурация
logging.basicConfig(level=logging.INFO)
CACHE_DIR = "cache"
DOCS_DIR = "docs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# API конфигурация (перенесено в глобальную область видимости)
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Настройки Streamlit
st.set_page_config(layout="wide", initial_sidebar_state="auto")
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, 
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, 
    .viewerBadge_text__1JaDK, #MainMenu, footer, header { 
        display: none !important; 
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Описание проекта")
st.sidebar.title("TEST-passer (AI-ассистент по тестам)")

class DocumentChunk:
    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.uploaded_files = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.doc_texts = []
    
    def split_text(self, text, max_tokens=2000):
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            tokens = tokenizer.tokenize(para)
            if len(tokenizer.tokenize(current_chunk + para)) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    chunks.append(para)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def load_pdf(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        chunks = self.split_text(page_text)
                        for chunk in chunks:
                            self.chunks.append(DocumentChunk(
                                text=chunk,
                                doc_name=file_name,
                                page_num=page_num + 1
                            ))
                            self.doc_texts.append(chunk)
                
                if self.chunks:
                    self.uploaded_files.append(file_name)
                    # Обновляем TF-IDF матрицу
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
                    return True
                else:
                    st.error(f"Не удалось извлечь текст из файла {file_name}")
                    return False
        except Exception as e:
            st.error(f"Ошибка загрузки PDF: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def find_most_relevant_chunks(self, query, top_k=3):
        """Находит наиболее релевантные чанки с помощью TF-IDF и косинусного сходства"""
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        return [(self.chunks[i].text, self.chunks[i].doc_name, self.chunks[i].page_num) 
                for i in top_indices if similarities[0][i] > 0.1]
    
    def get_document_names(self):
        return self.uploaded_files
    
    def load_with_cache(self):
        """Загружает документы из папки docs с использованием кэша"""
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        
        # Проверяем, есть ли документы для загрузки
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return
            
        # Проверяем кэш
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data['files'] == pdf_files:
                    self.chunks = cached_data['chunks']
                    self.uploaded_files = cached_data['files']
                    self.doc_texts = cached_data['doc_texts']
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
                    return
        
        # Загружаем документы
        for file_name in pdf_files:
            file_path = os.path.join(DOCS_DIR, file_name)
            with open(file_path, 'rb') as file:
                self.load_pdf(file.read(), file_name)
        
        # Сохраняем в кэш
        if self.chunks:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'files': self.uploaded_files,
                    'chunks': self.chunks,
                    'doc_texts': self.doc_texts
                }, f)

# Инициализация базы знаний
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()
    st.session_state.knowledge_base.load_with_cache()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Отображение загруженных документов
if st.session_state.knowledge_base.uploaded_files:  # Исправлено с loaded_files на uploaded_files
    st.subheader("📚 Используемые документы:")
    for doc in sorted(st.session_state.knowledge_base.uploaded_files):  # Аналогично
        st.markdown(f"- {doc}")
else:
    st.warning("В папке docs не найдено PDF-документов")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод вопроса
if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    relevant_chunks = st.session_state.knowledge_base.find_most_relevant_chunks(prompt)
    
    if not relevant_chunks:
        response_text = "Ответ не найден в материалах ❌"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):  # Опечатка исправлена на "assistant"
            st.markdown(response_text)
    else:
        context = "\n\n".join([f"Документ: {doc_name}, страница {page_num}\n{text}" 
                             for text, doc_name, page_num in relevant_chunks])
        
        full_prompt = f"""Ты — профессиональный бухгалтерский советник.
        Отвечай кратко, понятно и строго на основе следующих фрагментов нормативных актов (внизу указаны источники):

Question: {prompt}

Relevant materials:
{context}"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        with st.spinner("Ищем ответ..."):
            start_time = datetime.now()
            
            try:
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    full_response = response_data['choices'][0]['message']['content']
                    
                    sources = "\n\nИсточники:\n" + "\n".join(
                        [f"- {doc_name}, стр. {page_num}" for _, doc_name, page_num in relevant_chunks]
                    )
                    full_response += sources
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response + " ✅")
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.info(f"⏱️ Время обработки: {duration:.2f} сек")
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
                logging.error(f"API request error: {str(e)}")

# Кнопка очистки чата
if st.button("Очистить историю сообщений"):
    st.session_state.messages = []
    st.rerun()

# Кнопка обновления кэша
if st.button("Обновить кэш документов"):
    st.session_state.knowledge_base = KnowledgeBase()
    st.session_state.knowledge_base.load_with_cache()
    st.rerun()
