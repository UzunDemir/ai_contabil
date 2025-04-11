# ... (предыдущий код остается без изменений до класса KnowledgeBase)

class KnowledgeBase:
    # ... (предыдущие методы класса остаются без изменений)

    def add_document(self, file_content, file_name):
        """Добавляет новый документ в базу знаний"""
        if file_name in self.uploaded_files:
            st.warning(f"Документ '{file_name}' уже загружен")
            return False
        
        success = self.load_pdf(file_content, file_name)
        if success:
            self.update_cache()
        return success
    
    def update_cache(self):
        """Обновляет кэш базы знаний"""
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'files': self.uploaded_files,
                'chunks': self.chunks,
                'doc_texts': self.doc_texts
            }, f)
    
    def clear(self):
        """Очищает базу знаний"""
        self.chunks = []
        self.uploaded_files = []
        self.doc_texts = []
        self.tfidf_matrix = None
        cache_file = os.path.join(CACHE_DIR, "knowledge_base.cache")
        if os.path.exists(cache_file):
            os.remove(cache_file)

# ... (предыдущий код до раздела загрузки документов)

# Функционал загрузки новых документов
with st.sidebar.expander("📤 Загрузить новые документы"):
    uploaded_files = st.file_uploader(
        "Выберите PDF-документы", 
        type="pdf", 
        accept_multiple_files=True,
        help="Загрузите PDF-файлы для анализа"
    )
    
    if uploaded_files and st.button("Добавить документы"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Обработка {uploaded_file.name}..."):
                if st.session_state.knowledge_base.add_document(
                    uploaded_file.read(), 
                    uploaded_file.name
                ):
                    st.success(f"Документ '{uploaded_file.name}' успешно добавлен")
                else:
                    st.error(f"Ошибка при добавлении '{uploaded_file.name}'")
        st.rerun()

# Функционал управления базой знаний
with st.sidebar.expander("⚙️ Управление базой знаний"):
    if st.button("Очистить базу знаний"):
        st.session_state.knowledge_base.clear()
        st.success("База знаний очищена")
        st.rerun()
    
    st.download_button(
        label="Экспорт базы знаний",
        data=pickle.dumps(st.session_state.knowledge_base),
        file_name="knowledge_base.pkl",
        mime="application/octet-stream",
        help="Скачать текущую базу знаний"
    )

# Улучшенный интерфейс чата
st.title("📚 AI-ассистент по тестам")
st.caption("Загрузите документы и задавайте вопросы по их содержанию")

# Отображение загруженных документов с возможностью удаления
if st.session_state.knowledge_base.uploaded_files:
    with st.expander("📂 Загруженные документы", expanded=True):
        cols = st.columns([4, 1])
        cols[0].subheader("Название документа")
        cols[1].subheader("Действия")
        
        for doc in sorted(st.session_state.knowledge_base.uploaded_files):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"- {doc}")
            if col2.button("🗑️", key=f"del_{doc}", help="Удалить документ"):
                # Удаляем чанки связанные с этим документом
                st.session_state.knowledge_base.chunks = [
                    chunk for chunk in st.session_state.knowledge_base.chunks 
                    if chunk.doc_name != doc
                ]
                st.session_state.knowledge_base.uploaded_files.remove(doc)
                st.session_state.knowledge_base.doc_texts = [
                    text for i, text in enumerate(st.session_state.knowledge_base.doc_texts)
                    if st.session_state.knowledge_base.chunks[i].doc_name != doc
                ]
                
                # Обновляем TF-IDF матрицу
                if st.session_state.knowledge_base.doc_texts:
                    st.session_state.knowledge_base.tfidf_matrix = (
                        st.session_state.knowledge_base.vectorizer.fit_transform(
                            st.session_state.knowledge_base.doc_texts
                        )
                    )
                else:
                    st.session_state.knowledge_base.tfidf_matrix = None
                
                st.session_state.knowledge_base.update_cache()
                st.success(f"Документ '{doc}' удален")
                st.rerun()

# Улучшенный вывод релевантных фрагментов
def display_relevant_chunks(relevant_chunks):
    if not relevant_chunks:
        return
    
    with st.expander("🔍 Показать используемые фрагменты документов"):
        for i, (text, doc_name, page_num) in enumerate(relevant_chunks, 1):
            st.markdown(f"**Фрагмент #{i}** (из {doc_name}, стр. {page_num}):")
            st.text(text[:500] + "..." if len(text) > 500 else text)
            st.divider()

# Модифицированная обработка вопросов
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

Вопрос: {prompt}

Контекст:
{context}

Ответ должен быть структурированным и содержать:
1. Краткий вывод
2. Обоснование из документов
3. Точные ссылки на источники"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        with st.spinner("Анализирую документы..."):
            start_time = datetime.now()
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    response_data = response.json()
                    full_response = response_data['choices'][0]['message']['content']
                    
                    # Добавляем ссылки на документы
                    sources = "\n\n📚 **Источники:**\n" + "\n".join(
                        [f"- {doc_name}, стр. {page_num}" for _, doc_name, page_num in relevant_chunks]
                    )
                    full_response += sources
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response)
                        display_relevant_chunks(relevant_chunks)
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.toast(f"⏱️ Время обработки: {duration:.2f} сек", icon="⏱️")
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа от API")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
                logging.error(f"API request error: {str(e)}")

# ... (остальной код остается без изменений)
