from openai import OpenAI
from sentence_transformers import CrossEncoder
import arxiv
from termcolor import colored
import textwrap
import json

class ArXivAssistant:
    def __init__(self, llm_client): 
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = arxiv.Client()
        self.llm_client = llm_client
        self.session_memory = {
            'last_papers': [],
            'conversation_history': []
        }

    def _generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """объявляем квен"""
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen3-4B",  
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7, 
                max_tokens=max_tokens,
                timeout=20, 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {str(e)}")
            return ""
        
    def optimize_query(self, user_query: str) -> str:
        """генерируем кейворды для arXiv апи"""
        prompt = f"""
        Make a bunch of keyword by the query.
        Use only keywords, which are really in articles.
        Don't write not keywords! WRIYE ONLY KEYWORDS
        Query: Machine learning
        machine learning, AI, artificial intelligence, deep learning, 
        neural networks, algorithms, data science, supervised learning, 
        unsupervised learning, reinforcement learning, classification, 
        regression, clustering, natural language processing, NLP, 
        computer vision 
        Query: Biology
        biology, cell biology, molecular biology, genetics, DNA, RNA, 
        proteins, enzymes, metabolism, photosynthesis, cellular respiration, 
        mitosis, meiosis, chromosomes, genes, mutations, evolution, 
        natural selection, biodiversity, taxonomy, ecology, ecosystems, 
        food webs, biomes, microbiology, bacteria, viruses, fungi, immunology, 
        antibodies, vaccines, physiology, anatomy, zoology, botany, 
        plant biology, animal behavior, neuroscience, endocrinology, 
        hormones, homeostasis, biochemistry, carbohydrates, lipids 
        Query: {user_query}"""
        response = self._generate(prompt, max_tokens=512*8).strip().strip('"').split("</think>")
        return response[-1][:50].strip()
    
    def is_question_about_articles(self, query: str) -> int:
        """определение ТИПА контекстного вопроса"""
        prompt = f"""
    # Классификатор вопросов для научного ассистента
Текущий контекст:
- История диалога: {len(self.session_memory['conversation_history'])} сообщений

Отвечай ТОЛЬКО ОДНИМ ИЗ ТРЁХ ВАРИАНТОВ:
1. "answer" - если вопрос о ЗАГРУЖЕННЫХ статьях
2. "find" - если нужен чистый поиск
3. "hybrid" - если требуется и поиск, и анализ

## ПРАВИЛА КЛАССИФИКАЦИИ:
ЭТО ОЧЕНЬ ВАЖНО!!!!:КАК ТОЛЬКО ВИДИШЬ СЛОВА:"первая статья", "третье исследование","эти статьи","найденных статей" , "приведенных статей" и всех подобных-отвечай "answer"
### 1. "answer" (только для загруженных статей)
Используй, когда:
- Есть указание на позицию: "первая статья", "третье исследование","эти статьи","найденные статьи"
- Используются указательные местоимения: "эта работа", "тот метод"
- Вопрос о конкретных элементах из загруженных статей:
  • Авторы (если известны)
  • Методы/результаты
  • Рисунки/таблицы
  • Выводы
ЭТО ОЧЕНЬ ВАЖНО: Есть слова 'показанные','приведенные','найденные' и их синонимы во всех склонениях

Примеры:
"Кто автор второй статьи?" → answer
"Объясни метод из этой работы" → answer
"О чем говорится в приведенных статьях" → answer 
"Расскажи про найденные статьи" → answer
### 2. "find" (чистый поиск)
Используй для:
- Общих тем без специфики: "статьи по машинному обучению"
- Явных запросов на поиск: "найди исследования о..."

Примеры:
"Найди свежие работы по ИИ" → find
"Какие есть статьи по биологии?" → find
"химия" → find
"математика" → find

### 3. "hybrid" (поиск + анализ)
Используй когда:
- Запрос о конкретной статье, которой нет в контексте
- Нужно объяснить концепцию: "как работает..." и этого нет в контексте
- Требуется сравнение методов которых нет в контексте
- Вопрос требует синтеза информации
- "расскажи про что-то" и этого нет в контексте
ВАЖНО: ЭТОГО НЕ ДОЛЖНО БЫТЬ В КОНТЕКСТЕ

### КРИТИЧЕСКИЕ ПРАВИЛА ЭТО САМОЕ ГЛАВНОЕ:
1. Сначала проверь, есть ли в вопросе ссылки на загруженные статьи
2. Для общих объяснений и конкретных алгоритмов → всегда "hybrid"
3. Видишь слова "эти","найденные","приведенные","первые" и все их подобные и синонимы → всегда "answer"

Вопрос: {query}
Ответ: """
        
        answer = self._generate(prompt, max_tokens=10).strip().lower()
        if 'hybrid' in answer:
            return 2  # find and answer
        elif 'answer' in answer and 'find' not in answer:
            return 1  # answer
        else:
            return 0  # find
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "classify_query_type",
                "description": "Classify user request into three classes: answer (about loaded papers), find (new search), or hybrid (both).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_type": {  
                            "type": "string",
                            "enum": ["find", "answer", "hybrid"],
                            "description": """
                                 # Классификатор вопросов для научного ассистента
Текущий контекст:
- История диалога: {len(self.session_memory['conversation_history'])} сообщений

Отвечай ТОЛЬКО ОДНИМ ИЗ ТРЁХ ВАРИАНТОВ:
1. "answer" - если вопрос о ЗАГРУЖЕННЫХ статьях
2. "find" - если нужен чистый поиск
3. "hybrid" - если требуется и поиск, и анализ

## ПРАВИЛА КЛАССИФИКАЦИИ:
КАК ТОЛЬКО ВИДИШЬ СЛОВА:"первая статья", "третье исследование","эти статьи","найденных статей" , "приведенных статей" и всех подобных-отвечай "answer"
### 1. "answer" (только для загруженных статей)
Используй, когда:
- Есть указание на позицию: "первая статья", "третье исследование","эти статьи","найденные статьи"
- Используются указательные местоимения: "эта работа", "тот метод"
- Вопрос о конкретных элементах из загруженных статей:
  • Авторы (если известны)
  • Методы/результаты
  • Рисунки/таблицы
  • Выводы
ЭТО ОЧЕНЬ ВАЖНО: Есть слова 'показанные','приведенные','найденные' и их синонимы во всех склонениях

Примеры:
"Кто автор второй статьи?" → answer
"Объясни метод из этой работы" → answer
"О чем говорится в приведенных статьях" → answer 
"Расскажи про найденные статьи" → answer
### 2. "find" (чистый поиск)
Используй для:
- Общих тем без специфики: "статьи по машинному обучению"
- Явных запросов на поиск: "найди исследования о..."

Примеры:
"Найди свежие работы по ИИ" → find
"Какие есть статьи по биологии?" → find
"химия" → find
"математика" → find

### 3. "hybrid" (поиск + анализ)
Используй когда:
- Запрос о конкретной статье, которой нет в контексте
- Нужно объяснить концепцию: "как работает..." и этого нет в контексте
- Требуется сравнение методов которых нет в контексте
- Вопрос требует синтеза информации
- "расскажи про что-то" и этого нет в контексте
ВАЖНО: ЭТОГО НЕ ДОЛЖНО БЫТЬ В КОНТЕКСТЕ

### КРИТИЧЕСКИЕ ПРАВИЛА ЭТО САМОЕ ГЛАВНОЕ:
1. Сначала проверь, есть ли в вопросе ссылки на загруженные статьи
2. Для общих объяснений и конкретных алгоритмов → всегда "hybrid"
3. Видишь слова "эти","найденные","приведенные","первые" и все их подобные и синонимы → всегда "answer"

                            """
                        }
                    },
                    "required": ["query_type"] 
                }
            }
        }
    ]

    def classification(self, query: str) -> int:
        """Определение типа запроса через function calling"""
        try:
            response = self.llm_client.chat.completions.create(
                model = "Qwen/Qwen3-4B",
                messages = [{
                    "role": "user",
                    "content": f"Классифицируй запрос: {query}"
                }],
                tools = self.TOOLS, 
                tool_choice = {"type": "function", "function": {"name": "classify_query_type"}}
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            query_type = args["query_type"] 
            
            return {
                "answer": 1,
                "find": 0,
                "hybrid": 2
            }[query_type]
            
        except Exception as e:
            print(f"⚠️ Ошибка классификации: {e}")
            return 0  # fallback
    def answer_question_about_paper(self, question: str) -> str:
        """Отвечает на вопросы по найденным статьям"""
        if not self.session_memory['last_papers']:
            return "Поиск подходящих статей и ответов по ним."
        
        # контекст
        papers_context = "\n\n".join(
            f"Paper {idx+1}:\nTitle: {p.title}\nSummary: {p.summary}"
            for idx, p in enumerate(self.session_memory['last_papers'])
        )
        
        prompt = f"""
        You're a research assistant. Answer in English with:
        
        - 4-space indent for wrapped lines
        - Clear section breaks
        - Bullet points for lists
        
        Papers:
        {papers_context}
        
        Question: {question}
        
        Example response format:
        '''
        Key findings from Paper 1:
        
        • Proposed novel architecture that
          improves upon previous work
        
        • Key advantages:
          - 30% faster training
          - Better gradient flow
          - Simpler implementation
        '''
        
        Answer:
        """
        
        return self._generate(prompt, max_tokens=2048*2).split("</think>")[-1]
    
    def _format_conversation_history(self) -> str:
        """форматирование истории последние 3 запоминаем"""
        return "\n".join(
            f"User: {q}\nAssistant: {a}" 
            for q, a in self.session_memory['conversation_history'][-3:]
        )
    
    def _update_conversation_history(self, question: str, answer: str):
        """обновление истории"""
        self.session_memory['conversation_history'].append((question, answer))
    def search_arxiv(self, query: str, top_k: int = 5) -> list:
        """Поиск и ранжирование статей"""
        search_query = f"ti:{query} OR abs:{query}"  # ищем в заголовках и абстрактах
        search = arxiv.Search(
        query=search_query,
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
        )
        
        papers = list(self.client.results(search))
        # Ранжируем с помощью CrossEncoder
        paper_info = [f"{p.title} {p.summary}" for p in papers]
        if not papers:
            return []
        
        paper_info = [f"{p.title} {p.summary}" for p in papers]
        scores = self.reranker.predict([(query, text) for text in paper_info])
        ranked_papers = [p for _, p in sorted(zip(scores, papers), reverse=True)]
        self.session_memory['last_papers'] = ranked_papers[:top_k]  # <- Сохраняем!
        return ranked_papers[:top_k]  # <- Возвращаем для использования в run()

    def format_results(self, papers: list) -> str:
        """Форматирование результатов"""
        papers_str = "\n\n".join( 
            f"Title: {p.title}\n\nSummary: {p.summary}\n\n----------------------------------------------------" 
            for p in papers 
        )
        return papers_str

    def run(self):
        print("Научный ассистент с vLLM (для выхода введите 'q')")
        while True:
            user_query = input("\n💬 Ваш запрос: ").strip()
            if user_query.lower() in ('q', 'quit'):
                break

            try:
                has_papers = len(self.session_memory['last_papers']) > 0
                if self.classification(user_query) == 1:  # answer
                    if not has_papers:
                        print("❌ Нет загруженных статей. Сначала выполните поиск.")
                        continue
                    answer = self.answer_question_about_paper(user_query)
                    print(answer)
                    self._update_conversation_history(user_query, answer)
                elif self.classification(user_query) == 0:
                    optimized_query = self.optimize_query(user_query)
                    print(f"🔍 Оптимизированный запрос: {optimized_query}")
                    print("⌛ Ищу статьи на arXiv...")
                    papers = self.search_arxiv(optimized_query)
                    print(len(papers))
                    if papers:
                        print("\n📚 Результаты:")
                        formatted = self.format_results(papers)
                        print(formatted)
                    else:
                        print("❌ Статей не найдено")
                    continue
                else:
                    print('Пытаюсь ответить на ваш вопрос...')
                    optimized_query = self.optimize_query(user_query)
                    papers = self.search_arxiv(optimized_query)
                    if papers:
                        answer = self.answer_question_about_paper(user_query)
                        print(answer)
                        self._update_conversation_history(user_query, answer) 
                    else:
                        print('Ассистент затрудняется ответить на ваш вопрос...')

            except Exception as e:
                print(f"⚠️ Ошибка: {str(e)}")