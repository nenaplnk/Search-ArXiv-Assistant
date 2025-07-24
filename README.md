# ArXiv Research Assistant with vLLM

A Python-powered research assistant that uses vLLM and arXiv API to help you find relevant scientific papers.

## Prerequisites

- NVIDIA GPU (T4/A100 recommended)
- Python 3.9+
- CUDA 11.8
## Features

- 🔍 Semantic search of arXiv papers with query optimization
- 📝 Automatic summarization of technical papers
- 💬 Context-aware Q&A about retrieved papers
- ⚡ Accelerated inference with vLLM backend
- 🧠 Conversation memory for follow-up questions

## Quick Start
1. Install requirements:
```bash
pip install -r requirements.txt
python main.py
```
2. Start vllm server (file sc.sh)

   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --download-dir "/content/models" > /content/vllm_debug.log 2>&1 &
   ```
4. Run the assistant
   ```bash
   python main.py
   ```
5. Run the benchmark
   ```bash
   python benchmark.py
   ```
5. Example interactions
   💬 Ваш запрос: machine learning
🔍 Оптимизированный запрос: machine learning AI artificial intelligence deep learning neural networks algorithms data science supervised learning unsupervised learning reinforcement learning classification regression clustering natural language processing NLP computer vision
⌛ Ищу статьи на arXiv...

📚 Результаты:
Title: AI Learning Algorithms: Deep Learning, Hybrid Models, and Large-Scale Model Integration

Summary: In this paper, we discuss learning algorithms and their importance in
different types of applications which includes training to identify important
patterns and features in a straightforward, easy-to-understand manner. We will
review the main concepts of artificial intelligence (AI), machine learning
(ML), deep learning (DL), and hybrid models. Some important subsets of Machine
Learning algorithms such as supervised, unsupervised, and reinforcement
learning are also discussed in this paper. These techniques can be used for
some important tasks like prediction, classification, and segmentation.
Convolutional Neural Networks (CNNs) are used for image and video processing
and many more applications. We dive into the architecture of CNNs and how to
integrate CNNs with ML algorithms to build hybrid models. This paper explores
the vulnerability of learning algorithms to noise, leading to
misclassification. We further discuss the integration of learning algorithms
with Large Language Models (LLM) to generate coherent responses applicable to
many domains such as healthcare, marketing, and finance by learning important
patterns from large volumes of data. Furthermore, we discuss the next
generation of learning algorithms and how we may have an unified Adaptive and
Dynamic Network to perform important tasks. Overall, this article provides
brief overview of learning algorithms, exploring their current state,
applications and future direction.

----------------------------------------------------

Title: Continual Learning: Tackling Catastrophic Forgetting in Deep Neural Networks with Replay Processes

Summary: Humans learn all their life long. They accumulate knowledge from a sequence
of learning experiences and remember the essential concepts without forgetting
what they have learned previously. Artificial neural networks struggle to learn
similarly. They often rely on data rigorously preprocessed to learn solutions
to specific problems such as classification or regression. In particular, they
forget their past learning experiences if trained on new ones. Therefore,
artificial neural networks are often inept to deal with real-life settings such
as an autonomous-robot that has to learn on-line to adapt to new situations and
overcome new problems without forgetting its past learning-experiences.
Continual learning (CL) is a branch of machine learning addressing this type of
problem. Continual algorithms are designed to accumulate and improve knowledge
in a curriculum of learning-experiences without forgetting. In this thesis, we
propose to explore continual algorithms with replay processes. Replay processes
gather together rehearsal methods and generative replay methods. Generative
Replay consists of regenerating past learning experiences with a generative
model to remember them. Rehearsal consists of saving a core-set of samples from
past learning experiences to rehearse them later. The replay processes make
possible a compromise between optimizing the current learning objective and the
past ones enabling learning without forgetting in sequences of tasks settings.
We show that they are very promising methods for continual learning. Notably,
they enable the re-evaluation of past data with new knowledge and the
confrontation of data from different learning-experiences. We demonstrate their
ability to learn continually through unsupervised learning, supervised learning
and reinforcement learning tasks.

----------------------------------------------------

Title: Deep-Net: Deep Neural Network for Cyber Security Use Cases

Summary: Deep neural networks (DNNs) have witnessed as a powerful approach in this
year by solving long-standing Artificial intelligence (AI) supervised and
unsupervised tasks exists in natural language processing, speech processing,
computer vision and others. In this paper, we attempt to apply DNNs on three
different cyber security use cases: Android malware classification, incident
detection and fraud detection. The data set of each use case contains real
known benign and malicious activities samples. The efficient network
architecture for DNN is chosen by conducting various trails of experiments for
network parameters and network structures. The experiments of such chosen
efficient configurations of DNNs are run up to 1000 epochs with learning rate
set in the range [0.01-0.5]. Experiments of DNN performed well in comparison to
the classical machine learning algorithms in all cases of experiments of cyber
security use cases. This is due to the fact that DNNs implicitly extract and
build better features, identifies the characteristics of the data that lead to
better accuracy. The best accuracy obtained by DNN and XGBoost on Android
malware classification 0.940 and 0.741, incident detection 1.00 and 0.997 fraud
detection 0.972 and 0.916 respectively.

----------------------------------------------------

Title: The Next Big Thing(s) in Unsupervised Machine Learning: Five Lessons from Infant Learning

Summary: After a surge in popularity of supervised Deep Learning, the desire to reduce
the dependence on curated, labelled data sets and to leverage the vast
quantities of unlabelled data available recently triggered renewed interest in
unsupervised learning algorithms. Despite a significantly improved performance
due to approaches such as the identification of disentangled latent
representations, contrastive learning, and clustering optimisations, the
performance of unsupervised machine learning still falls short of its
hypothesised potential. Machine learning has previously taken inspiration from
neuroscience and cognitive science with great success. However, this has mostly
been based on adult learners with access to labels and a vast amount of prior
knowledge. In order to push unsupervised machine learning forward, we argue
that developmental science of infant cognition might hold the key to unlocking
the next generation of unsupervised learning approaches. Conceptually, human
infant learning is the closest biological parallel to artificial unsupervised
learning, as infants too must learn useful representations from unlabelled
data. In contrast to machine learning, these new representations are learned
rapidly and from relatively few examples. Moreover, infants learn robust
representations that can be used flexibly and efficiently in a number of
different tasks and contexts. We identify five crucial factors enabling
infants' quality and speed of learning, assess the extent to which these have
already been exploited in machine learning, and propose how further adoption of
these factors can give rise to previously unseen performance levels in
unsupervised learning.

----------------------------------------------------

Title: Patients' Severity States Classification based on Electronic Health Record (EHR) Data using Multiple Machine Learning and Deep Learning Approaches

Summary: This research presents an examination of categorizing the severity states of
patients based on their electronic health records during a certain time range
using multiple machine learning and deep learning approaches. The suggested
method uses an EHR dataset collected from an open-source platform to categorize
severity. Some tools were used in this research, such as openRefine was used to
pre-process, RapidMiner was used for implementing three algorithms (Fast Large
Margin, Generalized Linear Model, Multi-layer Feed-forward Neural Network) and
Tableau was used to visualize the data, for implementation of algorithms we
used Google Colab. Here we implemented several supervised and unsupervised
algorithms along with semi-supervised and deep learning algorithms. The
experimental results reveal that hyperparameter-tuned Random Forest
outperformed all the other supervised machine learning algorithms with 76%
accuracy as well as Generalized Linear algorithm achieved the highest precision
score 78%, whereas the hyperparameter-tuned Hierarchical Clustering with 86%
precision score and Gaussian Mixture Model with 61% accuracy outperformed other
unsupervised approaches. Dimensionality Reduction improved results a lot for
most unsupervised techniques. For implementing Deep Learning we employed a
feed-forward neural network (multi-layer) and the Fast Large Margin approach
for semi-supervised learning. The Fast Large Margin performed really well with
a recall score of 84% and an F1 score of 78%. Finally, the Multi-layer
Feed-forward Neural Network performed admirably with 75% accuracy, 75%
precision, 87% recall, 81% F1 score.

----------------------------------------------------

💬 Ваш запрос: tell me about first paper
'''
Key findings from Paper 1:

• Discusses learning algorithms and their importance in various applications, focusing on identifying patterns and features.

• Reviews AI, ML, DL, and hybrid models, explaining their main concepts.

• Explores supervised, unsupervised, and reinforcement learning as important subsets of ML algorithms.

• Introduces Convolutional Neural Networks (CNNs) for image and video processing, highlighting their architecture and integration with ML algorithms.

• Examines the vulnerability of learning algorithms to noise, leading to misclassification.

• Discusses the integration of learning algorithms with Large Language Models (LLMs) to generate coherent responses across multiple domains.

• Outlines the future of learning algorithms, suggesting the development of unified adaptive and dynamic networks.

• Provides a brief overview of learning algorithms, exploring their current state, applications, and future directions.
'''
