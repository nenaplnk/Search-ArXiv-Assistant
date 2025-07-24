%%bash --bg
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --download-dir "/content/models" > /content/vllm_debug.log 2>&1 &
echo "Сервер запущен. Логи: /content/vllm_debug.log"