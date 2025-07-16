from vllm import LLM, SamplingParams
from sentence_transformers import CrossEncoder
import arxiv
import textwrap
from datetime import datetime
from termcolor import colored

class ArXivAssistant:

    def __init__(self, model, sampling_params):
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = arxiv.Client()
        self.model = model
        self.sampling_params = sampling_params
        
    def optimize_query(self, user_query: str) -> str:
        """Генерация ключевых слов для arXiv с помощью vLLM"""
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
        outputs = self.model.generate(prompt, self.sampling_params)
        return " ".join([i.replace(",", "") for i in outputs[0].outputs[0].text.strip().split()])

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
        return ranked_papers[:top_k]

    def format_results(self, papers: list) -> str:
        """Форматирование результатов через vLLM"""
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
                # 1. Оптимизация запроса
                optimized_query = self.optimize_query(user_query)
                print(f"🔍 Оптимизированный запрос: {optimized_query}")

                # 2. Поиск статей
                print("⌛ Ищу статьи на arXiv...")
                papers = self.search_arxiv(optimized_query)
                # 3. Форматирование
                if papers:
                    print("\n📚 Результаты:")
                    formatted = self.format_results(papers)
                    print(formatted)
                else:
                    print("❌ Статей не найдено")

            except Exception as e:
                print(f"⚠️ Ошибка: {str(e)}")