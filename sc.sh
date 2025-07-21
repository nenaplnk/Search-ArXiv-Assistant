!python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct-GPTQ \
    --quantization gptq \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.7 \
    --download-dir /content/models 2>&1 | tee /content/vllm.log