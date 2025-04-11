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

# Конфигурация
CACHE_DIR = "cache"
DOCS_DIR = "docs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

#######
# Добавьте в начало файла (после импортов)
import logging
logging.basicConfig(level=logging.INFO)


st.sidebar.divider()
st.sidebar.write(
    """
    Это приложение использует предобработанные материалы из папки docs для быстрых ответов.
    Все документы автоматически кэшируются для ускорения работы.
    """
)

# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;
    }
    </style>
    <div class="center">
        <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">
        <h1>TEST-passer</h1>
        <h2>AI-ассистент по тестам</h2>
        <p>(строго по учебным материалам)</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Получение API ключа
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

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

# Где-то в интерфейсе (например, в сайдбаре)
with st.sidebar:
    if st.button("🛠 Тест кэша (DEBUG)"):
        try:
            st.info("Запуск теста кэширования...")
            
            # Создаем новую базу знаний для теста
            test_kb = KnowledgeBase()
            
            # Проверяем загрузку
            st.write("1. Проверка папок:")
            st.code(f"DOCS_DIR: {os.listdir(DOCS_DIR)}\nCACHE_DIR: {os.listdir(CACHE_DIR)}")
            
            # Тест обработки PDF
            st.write("2. Обработка документов:")
            test_kb.load_with_cache()
            
            # Проверка результатов
            st.write("3. Результаты:")
            if test_kb.chunks:
                st.success(f"✅ Успешно! Обработано {len(test_kb.chunks)} чанков")
                st.code(f"Последний чанк:\n{test_kb.chunks[-1].text[:200]}...")
            else:
                st.error("❌ Чанки не созданы!")
                
            # Показываем файлы кэша
            st.write("4. Содержимое cache/:")
            cache_files = os.listdir(CACHE_DIR)
            if cache_files:
                st.success(f"Найдены файлы кэша: {cache_files}")
                if "knowledge_base.cache" in cache_files:
                    st.code(f"Размер кэша: {os.path.getsize(os.path.join(CACHE_DIR, 'knowledge_base.cache'))} байт")
            else:
                st.error("Файлы кэша не найдены!")
                
        except Exception as e:
            st.error(f"Ошибка теста: {str(e)}")
            logging.exception("Ошибка в тесте кэша:")
            #################


    
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
    
    def process_pdf(self, file_path, file_name):
        try:
            with open(file_path, 'rb') as file:
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
                return True
        except Exception as e:
            st.error(f"Ошибка обработки PDF {file_name}: {e}")
            return False
    
    def build_vectorizer(self):
        if self.doc_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
    
    def find_most_relevant_chunks(self, query, top_k=3):
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        return [(self.chunks[i].text, self.chunks[i].doc_name, self.chunks[i].page_num) 
                for i in top_indices if similarities[0][i] > 0.1]
    
    def save_to_cache(self):
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'doc_texts': self.doc_texts,
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'loaded_files': self.loaded_files
            }, f)
        
        # Сохраняем хеш файлов
        hash_file = os.path.join(CACHE_DIR, "files_hash.txt")
        with open(hash_file, 'w') as f:
            f.write(self.get_files_hash())
    
    def load_from_cache(self):
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.doc_texts = data['doc_texts']
                    self.vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['tfidf_matrix']
                    self.loaded_files = data['loaded_files']
                return True
            except Exception as e:
                st.error(f"Ошибка загрузки кэша: {e}")
                return False
        return False
    
    def get_files_hash(self):
        """Вычисляет хеш всех PDF-файлов для проверки изменений"""
        hash_obj = hashlib.sha256()
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(DOCS_DIR, filename)
                with open(filepath, 'rb') as f:
                    while chunk := f.read(8192):
                        hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def load_with_cache(self):
        """Умная загрузка с проверкой хеша файлов"""
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        hash_file = os.path.join(CACHE_DIR, "files_hash.txt")
        
        current_hash = self.get_files_hash()
        
        # Если есть сохраненный хеш и он совпадает с текущим - загружаем из кеша
        if os.path.exists(hash_file) and os.path.exists(cache_file):
            with open(hash_file, 'r') as f:
                saved_hash = f.read().strip()
            
            if saved_hash == current_hash:
                if self.load_from_cache():
                    st.success("Загружены предобработанные данные из кэша")
                    return True
        
        # Если кеш невалиден - пересоздаем
        st.info("Обновление кэша документов...")
        self.chunks = []
        self.doc_texts = []
        self.loaded_files = set()
        
        for filename in os.listdir(DOCS_DIR):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(DOCS_DIR, filename)
                if self.process_pdf(file_path, filename):
                    self.loaded_files.add(filename)
                    st.success(f"Обработан документ: {filename}")
        
        self.build_vectorizer()
        self.save_to_cache()
        st.success("Кэш документов успешно обновлен!")
        return True

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

# Кнопка очистки чата
if st.button("Очистить историю сообщений"):
    st.session_state.messages = []
    st.rerun()

# Кнопка обновления кэша
if st.button("Обновить кэш документов"):
    st.session_state.knowledge_base = KnowledgeBase()
    st.session_state.knowledge_base.load_with_cache()
    st.rerun()
