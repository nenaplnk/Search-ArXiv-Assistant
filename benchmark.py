from model import ArXivAssistant
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
llm_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
t = ArXivAssistant(llm_client)
test_data = [
             {"запрос":"Mathematics","ответ":0},
             {"запрос":"расскажи про третью статью","ответ":1},
             {"запрос":"tell about gradient descent algorithm","ответ": 2},
             {"запрос":"названия предложенных статей","ответ":1},
             {"запрос":"physics","ответ": 0},
             {"запрос":"busik","ответ":0},
             {"запрос":"tell about attention algorithm","ответ":2},
             {"запрос":"сколько приведенных статей","ответ":1},
             {"запрос":"explain MMS algorithm","ответ":2},
             {"запрос":"MMS in machine learning","ответ":2},
             {"запрос":"explain integration in mathematics ","ответ":2},
             {"запрос":"show papers about recomender system","ответ":0},
             {"запрос":"про что вторая и первая статья","ответ":1},
             {"запрос":"methods of differentiation","ответ":2},
             {"запрос":"general theorem of algebra","ответ":0},
             {"запрос":"what kind of technology use in that papers","ответ":1},
             {"запрос":"Chemistry","ответ":0},
             {"запрос":"ML","ответ":0},
             {"запрос":"show summary of that paper","ответ":1},
             {"запрос":"Biology","ответ":0},
            ]
Y_true = []
Y_pred_fc = []
Y_pred_llm = [] 
for it in test_data:
    pred_llm = t.is_question_about_articles(it["запрос"])
    true = it["ответ"]
    pred_fc = t.classification(it["запрос"])
    Y_true.append(true)
    Y_pred_fc.append(pred_fc)
    Y_pred_llm.append(pred_llm)
accuracy_fc = accuracy_score(Y_true, Y_pred_fc)
precision_fc = precision_score(Y_true, Y_pred_fc, average='weighted')  
recall_fc = recall_score(Y_true, Y_pred_fc, average='weighted')
f1_fc = f1_score(Y_true, Y_pred_fc, average='weighted')
accuracy_llm = accuracy_score(Y_true, Y_pred_llm)
precision_llm = precision_score(Y_true, Y_pred_llm, average='weighted')  
recall_llm = recall_score(Y_true, Y_pred_llm, average='weighted')
f1_llm = f1_score(Y_true, Y_pred_llm, average='weighted')
print(f"""
      метрики классификации с llm : accuracy = {accuracy_llm:.2f}, precision = {precision_llm:.2f}, recall = {recall_llm}, f1 = {f1_llm:.2f}
      -----------------------------------------------------------------------------------------
      метрики с function calling: accuracy = {accuracy_fc:.2f}, precision = {precision_fc:.2f}, recall = {recall_fc} f1 = {f1_fc:.2f}
      """)