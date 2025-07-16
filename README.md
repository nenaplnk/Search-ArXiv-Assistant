# ArXiv Research Assistant with vLLM

A Python-powered research assistant that uses vLLM and arXiv API to help you find relevant scientific papers.

## Features

- 🔍 Smart query optimization using LLM (Qwen2-7B-Instruct)
- 📊 Relevance ranking with CrossEncoder
- 🚀 Fast search through arXiv's public API
- 📝 Clean terminal interface

## Quick Start
Requirements
Python 3.8+

GPU with at least 8GB VRAM (for vLLM)

Internet connection

Dependencies (auto-installed):

vllm - for LLM inference

arxiv - arXiv API client

sentence-transformers - for relevance ranking

Configuration
Edit main.py to change:

LLM model (Qwen/Qwen2-7B-Instruct)

Sampling parameters (temperature, max_tokens)

Limitations
Currently supports only arXiv (GitHub search coming soon)

Requires significant GPU resources
1. Install requirements:
```bash
pip install -r requirements.txt

