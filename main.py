from openai import OpenAI
from model import ArXivAssistant
from warnings import filterwarnings
filterwarnings("ignore")

if __name__ == "__main__":
    llm_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    assistant = ArXivAssistant(llm_client)  
    assistant.run()
