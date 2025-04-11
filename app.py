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
from io import BytesIO

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Настройки Streamlit
st.set_page_config(layout="wide", page_title="Бухгалтерский AI-советник Молдовы")
st.markdown("""
<style>
    .header {
        color: #2b5876;
        text-align: center;
        margin-bottom: 30px;
    }
    .law-badge {
        background-color: #f0f8ff;
        border-radius: 5px;
        padding: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Секреты API
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Законодательные акты Молдовы (URL PDF)
LAW_URLS = {
    "Закон о бухучете №113-XVI": "https://www.legis.md/cautare/downloadpdf/146721",
    "Налоговый кодекс": "https://www.legis.md/cautare/downloadpdf/143282",
    "Закон о финансовой отчетности": "https://www.legis.md/cautare/downloadpdf/137025",
    "Закон о налоге на прибыль": "https://www.legis.md/cautare/downloadpdf/142481",
    "Закон о НДС": "https://www.legis.md/cautare/downloadpdf/147850",
    "Трудовой кодекс": "https://www.legis.md/cautare/downloadpdf/131868"
}

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
    
    def load_pdf_from_url(self, url, doc_name):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with BytesIO(response.content) as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        chunks = self.split_text(page_text)
                        for chunk in chunks:
                            self.chunks.append(DocumentChunk(
                                text=chunk,
                                doc_name=doc_name,
                                page_num=page_num + 1
                            ))
                            self.doc_texts.append(chunk)
                
            if self.chunks:
                # Обновляем TF-IDF матрицу
                self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
                return True
            return False
            
        except Exception as e:
            st.error(f"Ошибка загрузки {doc_name}: {str(e)}")
            return False
    
    def find_relevant_chunks(self, query, top_k=3):
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        return [(self.chunks[i].text, self.chunks[i].doc_name, self.chunks[i].page_num) 
                for i in top_indices if similarities[0][i] > 0.1]

# Инициализация
if 'kb' not in st.session_state:
    st.session_state.kb = KnowledgeBase()
    # Автозагрузка законов при первом запуске
    with st.spinner("Загрузка законодательной базы..."):
        for name, url in LAW_URLS.items():
            if st.session_state.kb.load_pdf_from_url(url, name):
                st.success(f"Загружен: {name}")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Интерфейс
st.markdown("<h1 class='header'>Бухгалтерский AI-советник Молдовы</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #555;'>
Ответы строго на основании законодательства Республики Молдова<br>
(Налоговый кодекс, закон о бухучете, трудовое право)
</p>
""", unsafe_allow_html=True)

# Отображение загруженных документов
st.subheader("📜 Загруженные нормативные акты:")
for law in LAW_URLS.keys():
    st.markdown(f"<div class='law-badge'>{law}</div>", unsafe_allow_html=True)

# Чат
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Обработка вопросов
if prompt := st.chat_input("Ваш вопрос по бухучету/налогам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Поиск в документах
    relevant_chunks = st.session_state.kb.find_relevant_chunks(prompt)
    
    if not relevant_chunks:
        response = "Ответ не найден в законодательных актах ❌"
    else:
        # Формируем контекст
        context = "\n\n".join([f"📄 {doc_name} (ст. {page_num}):\n{text}" 
                             for text, doc_name, page_num in relevant_chunks])
        
        # Промпт для модели
        system_prompt = """Ты - экспертный помощник по бухгалтерскому учету и налогам Молдовы. Отвечай ТОЛЬКО на основании предоставленных нормативных актов. Если информация отсутствует - так и скажи.

Требования к ответу:
1. Ясный и простой язык
2. Только факты из документов
3. Обязательно укажи конкретные законы и статьи
4. Если вопрос не по теме - вежливо откажись отвечать

Документы:
{context}"""

        messages = [
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": prompt}
        ]
        
        with st.spinner("Анализирую законодательство..."):
            try:
                response = requests.post(url, headers=headers, json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }).json()
                
                answer = response['choices'][0]['message']['content']
                
                # Добавляем источники
                sources = "\n\n🔍 Источники:\n" + "\n".join(
                    f"- {doc_name} (ст. {page_num})" 
                    for _, doc_name, page_num in relevant_chunks
                )
                answer += sources
                
            except Exception as e:
                answer = f"Ошибка: {str(e)}"
    
    # Вывод ответа
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Кнопка очистки
if st.button("Очистить историю"):
    st.session_state.messages = []
    st.rerun()
