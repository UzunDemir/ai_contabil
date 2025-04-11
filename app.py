import os
import io
import json
import requests
import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔐 Установите ваш API-ключ в переменной окружения
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# 🔗 Список документов для загрузки
# LEGIS_URLS = [
#     "https://www.legis.md/cautare/downloadpdf/146721",
#     "https://www.legis.md/cautare/downloadpdf/143282",
#     "https://www.legis.md/cautare/downloadpdf/137025",
#     "https://www.legis.md/cautare/downloadpdf/142481",
#     "https://www.legis.md/cautare/downloadpdf/147850",
#     "https://www.legis.md/cautare/downloadpdf/131868"
# ]

LEGIS_URLS = [
    "https://www.legis.md/cautare/downloadpdf/146721",
    "https://www.legis.md/cautare/downloadpdf/143282",
    "https://www.legis.md/cautare/downloadpdf/137025",
    "https://www.legis.md/cautare/downloadpdf/142481",
    "https://www.legis.md/cautare/downloadpdf/147850",
    "https://www.legis.md/cautare/downloadpdf/131868"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

for url in urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        filename = url.split("/")[-1] + ".pdf"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Загружено: {filename}")
    except Exception as e:
        print(f"❌ Не удалось загрузить: {url} — {e}")



class KnowledgeBase:
    def __init__(self):
        self.text_chunks = []
        self.chunk_sources = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.uploaded_files = []

    def extract_text_from_pdf(self, file_content):
        text_pages = []
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_pages.append((text.strip(), i + 1))
            except Exception:
                continue
        return text_pages

    def split_into_chunks(self, text, max_tokens=500):
        sentences = text.split(". ")
        chunks = []
        current = []
        length = 0
        for sent in sentences:
            l = len(sent.split())
            if length + l > max_tokens:
                if current:
                    chunks.append(". ".join(current))
                    current = []
                    length = 0
            current.append(sent)
            length += l
        if current:
            chunks.append(". ".join(current))
        return chunks

    def load_pdf(self, file_content, file_name):
        pages = self.extract_text_from_pdf(file_content)
        for text, page_num in pages:
            chunks = self.split_into_chunks(text)
            for chunk in chunks:
                self.text_chunks.append(chunk)
                self.chunk_sources.append((file_name, page_num))
        self.uploaded_files.append(file_name)
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        if self.text_chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.text_chunks)

    def find_most_relevant_chunks(self, query, top_k=3):
        if not self.tfidf_matrix:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.text_chunks[i], *self.chunk_sources[i]) for i in top_indices]


@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase()


st.set_page_config(page_title="Советник бухгалтера", page_icon="📚")
st.title("📚 Советник бухгалтера")
st.caption("Отвечает строго на основе законодательства Республики Молдова")

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = get_knowledge_base()

# Загрузка документов
st.subheader("📥 Загрузка базы знаний")
for url in LEGIS_URLS:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split("/")[-1] + ".pdf"
            with st.spinner(f"Загрузка и обработка: {filename}"):
                kb = st.session_state.knowledge_base
                kb.load_pdf(response.content, filename)
                st.success(f"Загружено: {filename}")
        else:
            st.error(f"Не удалось загрузить: {url}")
    except Exception as e:
        st.error(f"Ошибка при загрузке {url}: {str(e)}")

# Отображение загруженных файлов
st.divider()
st.subheader("📄 Загруженные документы:")
if st.session_state.knowledge_base.uploaded_files:
    st.write("\n".join(f"- {name}" for name in st.session_state.knowledge_base.uploaded_files))
else:
    st.write("Документы пока не загружены.")

# Ввод вопроса
st.divider()
st.subheader("❓ Задайте вопрос")

user_input = st.text_input("Ваш вопрос по бухгалтерии:")
if user_input:
    kb = st.session_state.knowledge_base
    references = kb.find_most_relevant_chunks(user_input, top_k=3)

    if not references:
        st.warning("Не удалось найти релевантную информацию в загруженных документах.")
    else:
        context = "\n\n".join([f"(Источник: {doc}, стр. {page})\n{chunk}" for chunk, doc, page in references])

        prompt = f"""Ты — профессиональный бухгалтерский советник. Отвечай кратко, понятно и строго на основе следующих фрагментов нормативных актов (внизу указаны источники):

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{user_input}

ОТВЕТ:"""

        body = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        with st.spinner("Формируем ответ..."):
            response = requests.post(url, headers=headers, data=json.dumps(body))
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
                st.success("Ответ:")
                st.markdown(reply)
                st.markdown("---")
                for chunk, doc, page in references:
                    st.markdown(f"**Источник:** {doc}, стр. {page}")
            else:
                st.error(f"Ошибка API: {response.text}")
