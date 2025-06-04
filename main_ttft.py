#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import random
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from prompt import LONG_PROMPT_PAIRS, SHORT_PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TIMEOUT = 60 * 10


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class VLLMTTFTBenchmark:
    """A tool to benchmark vLLM service performance with focus on Time-to-First-Token optimization."""

    def __init__(
        self,
        vllm_url: str,
        initial_max_connections: int,
        duration: int,
        target_qps: float,
        api_key: Optional[str],
        model: str,
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.max_connections = initial_max_connections
        self.duration = duration
        self.target_qps = target_qps
        self.api_key = api_key
        self.model = model
        self.long_context = True

        # Metrics collection
        self.metrics_data: Dict[str, List[float]] = {
            "time": [],
            "ttft": [],  # Time to first token
            "running_requests": [],
            "qps": [],
            "token_processing_rate": [],
            "max_connections": [],
            "prompt_tokens_rate": [],  # Tokens per second for prompts
            "generation_tokens_rate": [],  # Tokens per second for generation
        }

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.last_token_count_time = None
        self.last_prompt_tokens = 0
        self.last_generation_tokens = 0

        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.start_time: Optional[float] = None
        self.stop_event = asyncio.Event()

        # TTFT tracking for gradual adjustments
        self.ttft_window = []  # Store last 10 TTFT measurements
        self.last_adjustment_time = 0
        self.adjustment_interval = 5
        self.adjustment_step = 10

        # Output directory
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)

    async def send_request(self, request_id: int) -> None:
        max_tokens = 1000
        if self.long_context:
            prompt_pair = random.choice(LONG_PROMPT_PAIRS)
            content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
            max_tokens = 9000
        else:
            content = " ".join((random.choice(SHORT_PROMPTS) * 100).split(" "))[:450]

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        self.active_requests += 1
        self.total_requests += 1
        request_start_time = time.time()
        first_token_received = False
        first_token_time = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT,
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    line_text = line.decode("utf-8").strip()
                                    if not line_text.startswith("data: "):
                                        continue

                                    json_str = line_text[6:]  # Remove "data: " prefix
                                    if json_str == "[DONE]":
                                        break

                                    data = json.loads(json_str)

                                    # Track token usage
                                    if "usage" in data:
                                        usage = data["usage"]
                                        if "prompt_tokens" in usage:
                                            self.total_prompt_tokens += usage[
                                                "prompt_tokens"
                                            ]
                                        if "completion_tokens" in usage:
                                            self.total_generation_tokens += usage[
                                                "completion_tokens"
                                            ]

                                    if not first_token_received:
                                        first_token_time = time.time()
                                        first_token_received = True
                                        # Record TTFT
                                        ttft = first_token_time - request_start_time
                                        self.metrics_data["ttft"].append(ttft)

                                        # Update TTFT window
                                        self.ttft_window.append(ttft)
                                        if len(self.ttft_window) > 10:
                                            self.ttft_window.pop(0)

                                        # Adjust max_connections periodically based on average TTFT
                                        current_time = time.time()
                                        if (
                                            current_time - self.last_adjustment_time
                                            >= self.adjustment_interval
                                        ):
                                            avg_ttft = sum(self.ttft_window) / len(
                                                self.ttft_window
                                            )
                                            if avg_ttft > 20:
                                                self.max_connections = max(
                                                    1,
                                                    self.max_connections
                                                    - self.adjustment_step,
                                                )
                                            else:
                                                self.max_connections += (
                                                    self.adjustment_step
                                                )

                                            self.last_adjustment_time = current_time
                                            self.metrics_data["max_connections"].append(
                                                self.max_connections
                                            )

                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON from stream: {line_text}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Error processing stream chunk: {e}"
                                    )

                        self.successful_requests += 1
                    else:
                        logger.error(f"Request failed: HTTP {response.status}")

        except asyncio.TimeoutError:
            logger.warning(f"Request timed out after {TIMEOUT} seconds")
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            # Log detailed exception information
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Request error: {error_type}: {error_msg}")
            logger.error(traceback.format_exc())
        finally:
            self.active_requests -= 1

    async def monitor_metrics(self) -> None:
        """Periodically collect and update metrics."""
        while not self.stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - self.start_time

            # Record metrics
            self.metrics_data["time"].append(elapsed)
            self.metrics_data["running_requests"].append(self.active_requests)

            # Calculate QPS (over last 5 seconds)
            if len(self.metrics_data["time"]) > 1:
                time_window = 5
                recent_requests = sum(
                    1 for t in self.metrics_data["time"] if t > elapsed - time_window
                )
                qps = recent_requests / time_window
                self.metrics_data["qps"].append(qps)

            # Calculate token processing rates
            if self.last_token_count_time is not None:
                time_diff = current_time - self.last_token_count_time
                if time_diff > 0:
                    prompt_tokens_rate = (
                        self.total_prompt_tokens - self.last_prompt_tokens
                    ) / time_diff
                    generation_tokens_rate = (
                        self.total_generation_tokens - self.last_generation_tokens
                    ) / time_diff

                    self.metrics_data["prompt_tokens_rate"].append(prompt_tokens_rate)
                    self.metrics_data["generation_tokens_rate"].append(
                        generation_tokens_rate
                    )

            self.last_token_count_time = current_time
            self.last_prompt_tokens = self.total_prompt_tokens
            self.last_generation_tokens = self.total_generation_tokens

            # Calculate average TTFT
            if self.metrics_data["ttft"]:
                avg_ttft = sum(self.metrics_data["ttft"][-10:]) / min(
                    10, len(self.metrics_data["ttft"])
                )
                logger.info(
                    f"Current metrics - Active requests: {self.active_requests}, "
                    f"Max connections: {self.max_connections}, "
                    f"Avg TTFT: {avg_ttft:.2f}s, "
                    f"QPS: {qps:.2f}, "
                    f"Prompt tokens/s: {prompt_tokens_rate:.2f}, "
                    f"Generation tokens/s: {generation_tokens_rate:.2f}"
                )

            await asyncio.sleep(1)

    async def generate_load(self) -> None:
        """Generate load with dynamic concurrency control."""
        interval = 1.0 / self.target_qps if self.target_qps > 0 else 0
        end_time = self.start_time + self.duration
        request_id = 0

        while time.time() < end_time and not self.stop_event.is_set():
            if self.active_requests < self.max_connections:
                asyncio.create_task(self.send_request(request_id))
                request_id += 1
                if interval > 0:
                    await asyncio.sleep(interval)
            else:
                await asyncio.sleep(0.1)  # Wait if at max connections

    def render_diagrams(self) -> None:
        """Render metrics as an HTML page with Plotly diagrams."""
        times = np.array(self.metrics_data["time"])
        if not times.size:
            logger.warning("No data to render")
            return

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Time to First Token",
                "Running Requests",
                "QPS",
                "Max Connections",
                "Prompt Tokens Rate",
                "Generation Tokens Rate",
            ),
            vertical_spacing=0.12,
        )

        # Add TTFT plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["ttft"],
                mode="lines+markers",
                name="TTFT",
                marker=dict(size=2),
            ),
            row=1,
            col=1,
        )

        # Add running requests plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["running_requests"],
                mode="lines+markers",
                name="Running Requests",
                marker=dict(size=2),
            ),
            row=1,
            col=2,
        )

        # Add QPS plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["qps"],
                mode="lines+markers",
                name="QPS",
                marker=dict(size=2),
            ),
            row=2,
            col=1,
        )

        # Add max connections plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["max_connections"],
                mode="lines+markers",
                name="Max Connections",
                marker=dict(size=2),
            ),
            row=2,
            col=2,
        )

        # Add prompt tokens rate plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["prompt_tokens_rate"],
                mode="lines+markers",
                name="Prompt Tokens/s",
                marker=dict(size=2),
            ),
            row=3,
            col=1,
        )

        # Add generation tokens rate plot
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["generation_tokens_rate"],
                mode="lines+markers",
                name="Generation Tokens/s",
                marker=dict(size=2),
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="vLLM TTFT Benchmark Results",
            showlegend=True,
            height=1600,
            width=1400,
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Seconds", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Requests/s", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Tokens/s", row=3, col=1)
        fig.update_yaxes(title_text="Tokens/s", row=3, col=2)

        output_file = os.path.join(
            self.output_dir, f"vllm_ttft_benchmark_{get_timestamp()}.html"
        )
        fig.write_html(output_file)
        logger.info(f"Benchmark results saved to {output_file}")

        # Print token usage statistics
        logger.info("\nToken Usage Statistics:")
        logger.info(f"Total Prompt Tokens: {self.total_prompt_tokens}")
        logger.info(f"Total Generation Tokens: {self.total_generation_tokens}")
        logger.info(
            f"Total Tokens: {self.total_prompt_tokens + self.total_generation_tokens}"
        )

    async def run(self) -> None:
        """Run the benchmark."""
        self.start_time = time.time()

        # Create tasks for monitoring metrics and generating load
        monitor_task = asyncio.create_task(self.monitor_metrics())
        load_task = asyncio.create_task(self.generate_load())

        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [monitor_task, load_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel the remaining task
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.render_diagrams()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="vLLM TTFT Benchmark Tool")
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--initial-max-connections",
        type=int,
        default=10,
        help="Initial maximum concurrent connections (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=10,
        help="Target queries per second (default: 10)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for vLLM authentication (optional)",
    )
    parser.add_argument(
        "--model",
        default="phala/deepseek-r1-70b",
        help="Model name for requests (default: phala/deepseek-r1-70b)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    benchmark = VLLMTTFTBenchmark(
        vllm_url=args.vllm_url,
        initial_max_connections=args.initial_max_connections,
        duration=args.duration,
        target_qps=args.qps,
        api_key=args.api_key,
        model=args.model,
    )

    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
