# vLLM Benchmark

This repository contains scripts for benchmarking the performance of large language models (LLMs) served using vLLM. It's designed to test the scalability and performance of LLM deployments under various concurrency levels.

## Requirements

- Python 3.10+
- Packages: `pip install -r requirements.txt`

## Usage

```bash
python3 main.py \
    --vllm-url <VLLM_URL> \
    --duration <DURATION> \
    --qps <QPS> \
    --initial-max-connections <INITIAL_MAX_CONNECTIONS> \
    --adjustment-step <ADJUSTMENT_STEP> \
    --max-connections <MAX_CONNECTIONS> \
    --api-key <API_KEY> \
    --model <MODEL>
```

Adjust the parameters of `--initial-max-connections`, `--adjustment-step` and `--max-connections` to get the best performance. Here is an example of the command:

```bash
python3 main_ttft.py \
    --vllm-url http://localhost:80 \
    --duration 600 \
    --qps 30 \
    --initial-max-connections 10 \
    --adjustment-step 10 \
    --max-connections 100 \
    --api-key TOKEN \
    --model deepseek/deepseek-chat-v3-0324
```