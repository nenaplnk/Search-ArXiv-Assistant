from openai import OpenAI
from sentence_transformers import CrossEncoder
import arxiv
from termcolor import colored
import textwrap


class ArXivAssistant:
    def __init__(self, llm_client):  # Убрать model и sampling_params
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
                model="Qwen/Qwen2-7B-Instruct",  # Модель жестко задана
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,  # Параметры задаются здесь
                max_tokens=max_tokens,
                timeout=20  # Таймаут для запроса
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {str(e)}")
            return ""
        
    def optimize_query(self, user_query: str) -> str:
        """генерируем кейворды для arXiv впи"""
        prompt = f"""
        Make a bunch of keyword by the query.
        Use only keywords, which are really in articles.
        Tag your answer with [ANSWER] and [/ANSWER]
        Query: Machine learning
        [ANSWER] machine learning, AI, artificial intelligence, deep learning, 
        neural networks, algorithms, data science, supervised learning, 
        unsupervised learning, reinforcement learning, classification, 
        regression, clustering, natural language processing, NLP, 
        computer vision [/ANSWER]
        Query: Biology
        [ANSWER] biology, cell biology, molecular biology, genetics, DNA, RNA, 
        proteins, enzymes, metabolism, photosynthesis, cellular respiration, 
        mitosis, meiosis, chromosomes, genes, mutations, evolution, 
        natural selection, biodiversity, taxonomy, ecology, ecosystems, 
        food webs, biomes, microbiology, bacteria, viruses, fungi, immunology, 
        antibodies, vaccines, physiology, anatomy, zoology, botany, 
        plant biology, animal behavior, neuroscience, endocrinology, 
        hormones, homeostasis, biochemistry, carbohydrates, lipids [/ANSWER]
        Query: {user_query}
        [ANSWER]
        """
        response = self._generate(prompt, max_tokens=100)
        
        keywords = response.strip().replace("\n", " ").replace(",", "")
        return " ".join(keywords.split())
    
    def is_question_about_articles(self, query: str) -> bool:
        """определение контекстного вопроса(y/n)"""
        prompt = f"""
    ### SYSTEM INSTRUCTIONS (STRICT):
    You are a classification bot that answers ONLY with 'yes' or 'no'.
    Your task is to determine if the user's question refers specifically to:
    - Research papers that were previously retrieved and displayed
    - Their content, authors, or metadata
    - Their order in the results (e.g., 'first paper', 'third article')
    
    ### RULES:
    1. Answer 'yes' ONLY if question explicitly references:
       - Papers by position (first/second/last)
       - Specific paper attributes (title, authors, methods)
       - Content from shown papers
    2. Answer 'no' for:
       - General knowledge questions
       - New search requests
       - Questions about fields of study
    
    ### DECISION TREE:
    IF question contains:
       - 'paper [number]' → yes
       - 'this/that/the [paper|article|study]' → yes
       - 'author(s) of' → yes
       - 'method in the' → yes
       - 'from the [results|papers]' → yes
    ELSE → no
    
    ### EXAMPLES (LEARN THESE PATTERNS):
    Q: What was in the first paper? → A: yes
    Q: Explain figure 3 from paper 2 → A: yes
    Q: Who wrote the last article? → A: yes
    Q: Summarize the methods → A: yes
    Q: What is deep learning? → A: no
    Q: Find papers about GANs → A: no
    Q: How does quantum computing work? → A: no
    
    ### FINAL INSTRUCTION:
    Analyze this question strictly using the rules above.
    You MUST answer with exactly one lowercase word:
    
    Question: {query}
    Answer: """
        
        answer = self._generate(prompt, max_tokens=10).strip().lower()
        return 'yes' in answer
    def answer_question_about_paper(self, question: str) -> str:
        """Отвечает на вопросы по найденным статьям"""
        if not self.session_memory['last_papers']:
            return "Осуществите поиск по статьям."
        
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
        
        return self._generate(prompt, max_tokens=2048)
    
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
        search = arxiv.Search(
            query=f"all:{query}",
            max_results=20,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = list(self.client.results(search))
        # Ранжируем с помощью CrossEncoder
        paper_info = [f"{p.title} {p.summary}" for p in papers]
        scores = self.reranker.predict([(query, text) for text in paper_info])

        # Сортируем по релевантности
        ranked_papers = [p for _, p in sorted(zip(scores, papers), reverse=True)]
        self.session_memory['last_papers'] = ranked_papers[:top_k]
        return ranked_papers[:top_k]

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
                print(self.is_question_about_articles(user_query)==1)
                has_papers = len(self.session_memory['last_papers']) > 0
                if self.is_question_about_articles(user_query) and has_papers:
                    answer = self.answer_question_about_paper(user_query)
                    print(answer)
                    self._update_conversation_history(user_query, answer)  # Здесь обновляем!
                    continue
                else:

                    optimized_query = self.optimize_query(user_query)
                    print(f"🔍 Оптимизированный запрос: {optimized_query}")
                    print("⌛ Ищу статьи на arXiv...")
                    papers = self.search_arxiv(optimized_query)
                    if papers:
                        print("\n📚 Результаты:")
                        formatted = self.format_results(papers)
                        print(formatted)
                    else:
                        print("❌ Статей не найдено")

            except Exception as e:
                print(f"⚠️ Ошибка: {str(e)}")
