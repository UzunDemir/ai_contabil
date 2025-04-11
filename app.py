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
import re

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

# ... (остальной код sidebar и заголовка оставляем без изменений)

class DocumentChunk:
    def __init__(self, text, doc_name, page_num, doc_title=None):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num
        self.doc_title = doc_title
        self.embedding = None

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.uploaded_files = []
        self.document_titles = {}  # Словарь для хранения заголовков документов
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.doc_texts = []
    
    def extract_title_from_text(self, text):
        """Извлекает заголовок из текста (первая строка с заглавными буквами)"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and line.isupper() and len(line) > 10 and len(line) < 150:
                return line
            # Попробуем найти шаблоны типа "ФЕДЕРАЛЬНЫЙ ЗАКОН" или "НАЛОГОВЫЙ КОДЕКС"
            if re.match(r'^(ФЕДЕРАЛЬНЫЙ|НАЛОГОВЫЙ|ГРАЖДАНСКИЙ|ТРУДОВОЙ|БЮДЖЕТНЫЙ)\s+(ЗАКОН|КОДЕКС)', line, re.IGNORECASE):
                return line
        return None
    
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
                doc_title = None
                
                # Сначала попробуем извлечь заголовок из первой страницы
                first_page_text = reader.pages[0].extract_text()
                doc_title = self.extract_title_from_text(first_page_text)
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        chunks = self.split_text(page_text)
                        for chunk in chunks:
                            self.chunks.append(DocumentChunk(
                                text=chunk,
                                doc_name=file_name,
                                page_num=page_num + 1,
                                doc_title=doc_title
                            ))
                            self.doc_texts.append(chunk)
                
                if self.chunks:
                    self.uploaded_files.append(file_name)
                    if doc_title:
                        self.document_titles[file_name] = doc_title
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
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            if not pdf_files:
                st.warning(f"В папке {folder_path} не найдено PDF-файлов")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_name in enumerate(pdf_files):
                file_path = os.path.join(folder_path, file_name)
                status_text.text(f"Загрузка {i+1}/{len(pdf_files)}: {file_name}...")
                
                with open(file_path, 'rb') as file:
                    file_content = file.read()
                    self.load_pdf(file_content, file_name)
                
                progress_bar.progress((i + 1) / len(pdf_files))
            
            progress_bar.empty()
            status_text.text("Загрузка завершена!")
            time.sleep(1)
            status_text.empty()
    
    # ... (остальные методы класса остаются без изменений)

# Инициализация
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()
    st.session_state.knowledge_base.load_from_folder()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Отображение загруженных документов с заголовками
if st.session_state.knowledge_base.get_document_names():
    st.subheader("📚 Загруженные документы:")
    for doc in st.session_state.knowledge_base.get_document_names():
        title = st.session_state.knowledge_base.document_titles.get(doc, doc)
        st.markdown(f"- {title}")
else:
    st.info("ℹ️ В папке 'legis' не найдено PDF-документов. Поместите файлы в эту папку.")

# ... (остальной код оставляем без изменений)

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
