import os
import pickle
import hashlib
import streamlit as st
import requests
import numpy as np
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
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.doc_texts = []
        self.loaded_files = set()
    
    # ... (остальные методы класса KnowledgeBase остаются без изменений)

# Инициализация базы знаний
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()
    st.session_state.knowledge_base.load_with_cache()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Отображение загруженных документов
if st.session_state.knowledge_base.loaded_files:
    st.subheader("📚 Используемые документы:")
    for doc in sorted(st.session_state.knowledge_base.loaded_files):
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
        with st.chat_message("assistant"):
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
