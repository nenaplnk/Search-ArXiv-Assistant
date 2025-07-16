from model import *
from warnings import filterwarnings
filterwarnings("ignore")
sampling_params = SamplingParams(temperature=0.3, max_tokens=200, stop=['[/ANSWER]'])
 
if __name__ == "__main__":
  llm = LLM(model="Qwen/Qwen2-7B-Instruct")
  assistant = ArXivAssistant(llm, sampling_params)
  assistant.run()
