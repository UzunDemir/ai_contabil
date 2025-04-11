import os
import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
from transformers import GPT2Tokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
st.sidebar.divider()
st.sidebar.write(
    """
    Это приложение выполнено в целях помощи студентам при сдаче тестов по ЛЮБОЙ образовательной теме.                  

    1. Как это работает? 
     
    Студент загружает учебный материал в pdf. TEST-passer отвечает на тесты, выбирая правильные ответы. 
     
    2. Почему не воспользоваться обычными чатами (GPT, DeepSeek и т. д.)? 
     
    Несмотря на то что модель обучена на огромном 
    количестве информации, она не понимает информацию, как человек, а лишь предсказывает "вероятный следующий фрагмент текста". 
    Она также имеет способность "галлюцинировать", то есть может "придумать" факт, источник или термин, которого не существует, но звучит правдоподобно.
    Поэтому наиболее правильные ответы будет выдавать модель, которая использует только НУЖНЫЙ материал. 
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
        <p> (строго по учебным материалам)</p>
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
        self.embedding = None

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
    
    def load_from_folder(self, folder_path="legis"):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.pdf'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'rb') as file:
                        file_content = file.read()
                        self.load_pdf(file_content, file_name)
    
    def find_most_relevant_chunks(self, query, top_k=3):
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        return [(self.chunks[i].text, self.chunks[i].doc_name, self.chunks[i].page_num) 
                for i in top_indices if similarities[0][i] > 0.1]
    
    def get_document_names(self):
        return self.uploaded_files

# Инициализация
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()
    # Загружаем файлы из папки legis при инициализации
    st.session_state.knowledge_base.load_from_folder()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Загрузка документов (дополнительно к файлам из папки legis)
uploaded_files = st.file_uploader("Загрузить дополнительные учебные материалы в PDF", type="pdf", accept_multiple_files=True)
if uploaded_files:
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name not in st.session_state.knowledge_base.get_document_names():
            success = st.session_state.knowledge_base.load_pdf(uploaded_file.getvalue(), uploaded_file.name)
            progress_bar.progress((i + 1) / len(uploaded_files))
            if success:
                st.success(f"Файл {uploaded_file.name} успешно загружен")

# Отображение загруженных документов
if st.session_state.knowledge_base.get_document_names():
    st.subheader("📚 Загруженные документы:")
    for doc in st.session_state.knowledge_base.get_document_names():
        st.markdown(f"- {doc}")
else:
    st.info("ℹ️ Документы не загружены. Поместите PDF-файлы в папку 'legis' или загрузите их через интерфейс.")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод вопроса
if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Поиск наиболее релевантных чанков
    relevant_chunks = st.session_state.knowledge_base.find_most_relevant_chunks(prompt)
    
    if not relevant_chunks:
        response_text = "Ответ не найден в материалах ❌"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        # Формируем контекст из релевантных чанков
        context = "\n\n".join([f"Документ: {doc_name}, страница {page_num}\n{text}" 
                             for text, doc_name, page_num in relevant_chunks])
        
        full_prompt = f"""Твоя роль - ассистент по бухгалтерскому учету. Отвечай строго на основании предоставленных нормативных документов по бухгалтерскому учету и налогообложению. 
Ответ должен быть простыми словами, но точным и содержать ссылки на конкретные статьи и пункты документов.
Если вопрос требует расчета - предоставь формулу и пример расчета.
Если ответ не найден в документах, ответь: 'Ответ не найден в нормативных документах

Educational materials: {prompt}

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
                    
                    # Добавляем ссылки на источники
                    sources = "\n\nИсточники:\n" + "\n".join(
                        [f"- {doc_name}, стр. {page_num}" for _, doc_name, page_num in relevant_chunks]
                    )
                    full_response += sources
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response + " ✅")
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.info(f"⏱️ Поиск ответа занял {duration:.2f} секунд")
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

# Кнопка очистки чата
if st.button("Очистить чат"):
    st.session_state.messages = []
    st.rerun()
