# vLLM Benchmark

This repository contains scripts for benchmarking the performance of large language models (LLMs) served using vLLM. It's designed to test the scalability and performance of LLM deployments under various concurrency levels.

## Requirements

- Python 3.10+
- Packages: `pip install -r requirements.txt`

## Usage

```bash
python3 main.py \
    --vllm-url <vLLM_URL> \
    --metrics-url <metrics_URL> \
    --duration <duration> \
    --qps <qps> \
    --concurrency <concurrency> \
    --api-key <your-api-key>
```
