#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class VLLMBenchmark:
    """A tool to benchmark vLLM service performance."""

    def __init__(
        self,
        vllm_url: str,
        metrics_url: str,
        duration: int,
        target_qps: float,
        concurrency: int,
        api_key: Optional[str],
        model: str,
        payload_config: Dict,
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.metrics_url = metrics_url.rstrip("/")
        self.duration = duration
        self.target_qps = target_qps
        self.concurrency = concurrency
        self.api_key = api_key
        if not api_key:
            logger.warning(
                "No API key provided; requests may fail if authentication is required"
            )
        self.model = model
        self.payload_config = payload_config
        self.metrics_data: Dict[str, List[float]] = {
            "time": [],
            "avg_prompt_tps": [],
            "avg_generation_tps": [],
            "running_requests": [],
            "pending_requests": [],
            "qps": [],
            "concurrent_requests": [],
            # Additional metrics
            "request_success": [],
            "time_to_first_token": [],
            "e2e_latency": [],
            "queue_time": [],
            "inference_time": [],
            "prefill_time": [],
            "decode_time": [],
            "finished_requests": [],  # New metric for finished request count
        }
        self.request_count = 0
        self.finished_request_count = 0  # Counter for finished requests
        self.metrics_session: Optional[aiohttp.ClientSession] = None
        self.load_session: Optional[aiohttp.ClientSession] = None
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        # For rate calculations
        self.prev_prompt_tokens = 0.0
        self.prev_generation_tokens = 0.0
        self.prev_time: Optional[float] = None
        self.stop_event = asyncio.Event()

    async def fetch_metrics(self) -> None:
        """Fetch and parse metrics from vLLM /metrics endpoint."""
        try:
            async with self.metrics_session.get(self.metrics_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch metrics: HTTP {response.status}")
                    return
                metrics_text = await response.text()
                timestamp = time.time()

                metrics = {
                    "prompt_tokens": None,
                    "generation_tokens": None,
                    "running": None,
                    "pending": None,
                    "request_success": None,
                    "time_to_first_token": None,
                    "e2e_latency": None,
                    "queue_time": None,
                    "inference_time": None,
                    "prefill_time": None,
                    "decode_time": None,
                }

                for family in text_string_to_metric_families(metrics_text):
                    for sample in family.samples:
                        # Only process metrics for the specified model
                        if (
                            "model_name" in sample.labels
                            and sample.labels["model_name"] != self.model
                        ):
                            continue

                        if sample.name == "vllm:prompt_tokens_total":
                            metrics["prompt_tokens"] = sample.value
                        elif sample.name == "vllm:generation_tokens_total":
                            metrics["generation_tokens"] = sample.value
                        elif sample.name == "vllm:num_requests_running":
                            metrics["running"] = sample.value
                        elif sample.name == "vllm:num_requests_waiting":
                            metrics["pending"] = sample.value
                        elif (
                            sample.name == "vllm:request_success_total"
                            and sample.labels.get("finished_reason") == "stop"
                        ):
                            metrics["request_success"] = sample.value
                        elif sample.name == "vllm:time_to_first_token_seconds_count":
                            metrics["time_to_first_token"] = sample.value
                        elif sample.name == "vllm:e2e_request_latency_seconds_count":
                            metrics["e2e_latency"] = sample.value
                        elif sample.name == "vllm:request_queue_time_seconds_count":
                            metrics["queue_time"] = sample.value
                        elif sample.name == "vllm:request_inference_time_seconds_count":
                            metrics["inference_time"] = sample.value
                        elif sample.name == "vllm:request_prefill_time_seconds_count":
                            metrics["prefill_time"] = sample.value
                        elif sample.name == "vllm:request_decode_time_seconds_count":
                            metrics["decode_time"] = sample.value

                # Calculate rates for tokens
                prompt_tps = generation_tps = 0.0
                if self.prev_time and timestamp > self.prev_time:
                    time_diff = timestamp - self.prev_time
                    if (
                        metrics["prompt_tokens"] is not None
                        and self.prev_prompt_tokens is not None
                    ):
                        prompt_tps = (
                            metrics["prompt_tokens"] - self.prev_prompt_tokens
                        ) / time_diff
                    if (
                        metrics["generation_tokens"] is not None
                        and self.prev_generation_tokens is not None
                    ):
                        generation_tps = (
                            metrics["generation_tokens"] - self.prev_generation_tokens
                        ) / time_diff

                # Update metrics data
                self.metrics_data["time"].append(timestamp)
                self.metrics_data["avg_prompt_tps"].append(max(0, prompt_tps))
                self.metrics_data["avg_generation_tps"].append(max(0, generation_tps))

                # Update running and pending requests
                if metrics["running"] is not None:
                    self.metrics_data["running_requests"].append(metrics["running"])
                else:
                    self.metrics_data["running_requests"].append(0)

                if metrics["pending"] is not None:
                    self.metrics_data["pending_requests"].append(metrics["pending"])
                else:
                    self.metrics_data["pending_requests"].append(0)

                # Update finished request count from client-side counter
                self.metrics_data["finished_requests"].append(
                    self.finished_request_count
                )

                # Update time to first token metric
                if metrics["time_to_first_token"] is not None:
                    self.metrics_data["time_to_first_token"].append(metrics["time_to_first_token"])
                else:
                    self.metrics_data["time_to_first_token"].append(0)

                # Update prefill time metric
                if metrics["prefill_time"] is not None:
                    self.metrics_data["prefill_time"].append(metrics["prefill_time"])
                else:
                    self.metrics_data["prefill_time"].append(0)

                # Update other latency metrics
                for metric_name in ["e2e_latency", "queue_time", "inference_time", "decode_time"]:
                    if metrics[metric_name] is not None:
                        self.metrics_data[metric_name].append(metrics[metric_name])
                    else:
                        self.metrics_data[metric_name].append(0)

                # Update previous values for rate calculations
                if metrics["prompt_tokens"] is not None:
                    self.prev_prompt_tokens = metrics["prompt_tokens"]
                if metrics["generation_tokens"] is not None:
                    self.prev_generation_tokens = metrics["generation_tokens"]
                self.prev_time = timestamp

                # Log additional metrics if available
                if metrics["request_success"] is not None:
                    logger.info(f"Successful requests: {metrics['request_success']}")
                if metrics["time_to_first_token"] is not None:
                    logger.info(
                        f"Time to first token requests: {metrics['time_to_first_token']}"
                    )
                if metrics["e2e_latency"] is not None:
                    logger.info(
                        f"End-to-end latency requests: {metrics['e2e_latency']}"
                    )
                if metrics["queue_time"] is not None:
                    logger.info(f"Queue time requests: {metrics['queue_time']}")
                if metrics["inference_time"] is not None:
                    logger.info(f"Inference time requests: {metrics['inference_time']}")
                if metrics["prefill_time"] is not None:
                    logger.info(f"Prefill time requests: {metrics['prefill_time']}")
                if metrics["decode_time"] is not None:
                    logger.info(f"Decode time requests: {metrics['decode_time']}")

        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")

    async def send_request(self) -> None:
        """Send a single request to vLLM API with optional API key authentication."""
        try:
            payload = {
                "model": self.model,
                "messages": self.payload_config.get(
                    "messages", [{"role": "user", "content": "Tell me about AI."}]
                ),
                "max_tokens": self.payload_config.get("max_tokens", 100),
            }
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            async with self.load_session.post(
                f"{self.vllm_url}/v1/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    self.request_count += 1
                    self.finished_request_count += (
                        1  # Increment finished request counter
                    )
                elif response.status == 401:
                    logger.error("Authentication failed: Invalid or missing API key")
                else:
                    logger.error(
                        f"Request failed: HTTP {response.status}, {await response.text()}"
                    )
        except Exception as e:
            logger.error(f"Request error: {e}")

    def _should_stop_early(self, check_tasks: bool = False) -> bool:
        """Check if the benchmark should stop early because all requests are completed.

        Args:
            check_tasks: If True, also check if there are no pending tasks.

        Returns:
            bool: True if the benchmark should stop early, False otherwise.
        """
        # Only check after 60 seconds
        if time.time() - self.start_time <= 60:
            return False

        # Check if all requests are completed
        if not self.metrics_data["running_requests"]:
            return False
        if (
            self.metrics_data["running_requests"][-1] != 0
            or self.metrics_data["pending_requests"][-1] != 0
        ):
            return False

        # Optionally check if there are no pending tasks
        if check_tasks and len(self.tasks) > 0:
            return False

        # Record finish time if not already set
        if self.finish_time is None:
            self.finish_time = time.time()
            logger.info(f"All requests completed at {self.finish_time:.2f} seconds")
            logger.info(
                f"Total time to complete all requests: {self.finish_time - self.start_time:.2f} seconds"
            )

        return True

    async def monitor_metrics(self) -> None:
        """Periodically fetch metrics during the benchmark."""
        while (
            time.time() - self.start_time < self.duration
            and not self.stop_event.is_set()
        ):
            await self.fetch_metrics()
            logger.info(
                f"Metrics fetched at {time.time() - self.start_time:.2f} seconds"
            )
            await asyncio.sleep(1)

            # Check if all requests have been completed - only stop if we're near the end of the duration
            if self._should_stop_early():
                logger.info(
                    "All requests have been completed. Stopping benchmark early."
                )
                self.stop_event.set()
                break

    async def generate_load(self) -> None:
        """Generate load at target QPS with concurrency control."""
        interval = 1.0 / self.target_qps if self.target_qps > 0 else 0
        end_time = self.start_time + self.duration
        self.tasks = []

        logger.info("Warming up...")
        await asyncio.sleep(10)  # wait for 10 seconds to warm up
        while time.time() < end_time and not self.stop_event.is_set():
            # Check if all requests have been completed - only stop if we're near the end of the duration
            if self._should_stop_early(check_tasks=True):
                logger.info(
                    "All requests have been completed. Stopping load generation early."
                )
                self.stop_event.set()
                break

            if len(self.tasks) < self.concurrency:
                self.tasks.append(asyncio.create_task(self.send_request()))
                if interval > 0:
                    await asyncio.sleep(interval)
            else:
                done, pending = await asyncio.wait(
                    self.tasks, timeout=interval, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = list(pending) + [t for t in done if not t.exception()]

        if self.tasks:
            await asyncio.wait(self.tasks)

        logger.info("Cooling down...")
        await asyncio.sleep(10)  # wait for 10 seconds to cool down

    def calculate_metrics(self) -> None:
        """Calculate QPS and concurrent requests per second."""
        times = np.array(self.metrics_data["time"])
        if len(times) < 2:
            logger.warning("Not enough data to calculate metrics")
            return

        elapsed = times[-1] - times[0]

        # Calculate QPS based on successful requests if available
        if (
            self.metrics_data["request_success"]
            and len(self.metrics_data["request_success"]) > 0
        ):
            total_requests = self.metrics_data["request_success"][-1]
            qps = total_requests / elapsed if elapsed > 0 else 0
        else:
            # Fallback to request_count if request_success is not available
            qps = self.request_count / elapsed if elapsed > 0 else 0

        # Calculate average QPS for each time point
        for _ in range(len(times)):
            self.metrics_data["qps"].append(qps)
            self.metrics_data["concurrent_requests"].append(
                self.metrics_data["running_requests"][_]
            )

        # Log summary metrics
        logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
        logger.info(f"Average QPS: {qps:.2f}")

        # Log latency metrics if available
        if (
            self.metrics_data["time_to_first_token"]
            and len(self.metrics_data["time_to_first_token"]) > 0
        ):
            logger.info(
                f"Total time to first token requests: {self.metrics_data['time_to_first_token'][-1]}"
            )

        if (
            self.metrics_data["prefill_time"]
            and len(self.metrics_data["prefill_time"]) > 0
        ):
            logger.info(
                f"Total prefill time requests: {self.metrics_data['prefill_time'][-1]}"
            )

        if (
            self.metrics_data["decode_time"]
            and len(self.metrics_data["decode_time"]) > 0
        ):
            logger.info(
                f"Total decode time requests: {self.metrics_data['decode_time'][-1]}"
            )

        if (
            self.metrics_data["e2e_latency"]
            and len(self.metrics_data["e2e_latency"]) > 0
        ):
            logger.info(
                f"Total end-to-end latency requests: {self.metrics_data['e2e_latency'][-1]}"
            )

        if self.metrics_data["queue_time"] and len(self.metrics_data["queue_time"]) > 0:
            logger.info(
                f"Total queue time requests: {self.metrics_data['queue_time'][-1]}"
            )

        if (
            self.metrics_data["inference_time"]
            and len(self.metrics_data["inference_time"]) > 0
        ):
            logger.info(
                f"Total inference time requests: {self.metrics_data['inference_time'][-1]}"
            )

        # Log finished request count if available
        if (
            self.metrics_data["finished_requests"]
            and len(self.metrics_data["finished_requests"]) > 0
        ):
            logger.info(
                f"Total finished requests (client-side): {self.metrics_data['finished_requests'][-1]}"
            )

    def render_diagrams(self) -> None:
        """Render metrics as an HTML page with Plotly diagrams."""
        times = np.array(self.metrics_data["time"])
        if not times.size:
            logger.warning("No data to render")
            return
        times = times - times[0]

        # Create a more comprehensive visualization with additional metrics
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Token Processing Rate",
                "Running Requests",
                "Pending Requests",
                "QPS",
                "Concurrent Requests/s",
                "Request Success Rate",
                "Latency Metrics",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Add combined token metrics
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["avg_prompt_tps"],
                mode="lines+markers",
                name="Prompt Tokens/s",
                marker=dict(size=4)
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["avg_generation_tps"],
                mode="lines+markers",
                name="Generation Tokens/s",
                marker=dict(size=4)
            ),
            row=1,
            col=1,
        )

        # Rearranged metrics
        metrics = [
            ("running_requests", "Running Requests", 1, 2),
            ("pending_requests", "Pending Requests", 2, 1),
            ("qps", "QPS", 2, 2),
            ("concurrent_requests", "Concurrent Requests/s", 3, 1),
        ]

        for key, title, row, col in metrics:
            fig.add_trace(
                go.Scatter(
                    x=times, y=self.metrics_data[key], mode="lines+markers", name=title,
                    marker=dict(size=4)
                ),
                row=row,
                col=col,
            )

        # Add request success rate if available
        if (
            "request_success" in self.metrics_data
            and self.metrics_data["request_success"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.metrics_data["request_success"],
                    mode="lines+markers",
                    name="Successful Requests",
                    marker=dict(size=4)
                ),
                row=3,
                col=2,
            )

        # Add finished request count if available
        if (
            "finished_requests" in self.metrics_data
            and self.metrics_data["finished_requests"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.metrics_data["finished_requests"],
                    mode="lines+markers",
                    name="Finished Requests",
                    marker=dict(size=4)
                ),
                row=3,
                col=2,
            )

        # Add latency metrics if available
        latency_metrics = [
            ("time_to_first_token", "Time to First Token (s)", 4, 1),
            ("prefill_time", "Prefill Time (s)", 4, 1),
            ("decode_time", "Decode Time (s)", 4, 1),
            ("e2e_latency", "End-to-End Latency (s)", 4, 1),
            ("queue_time", "Queue Time (s)", 4, 1),
            ("inference_time", "Inference Time (s)", 4, 1),
        ]

        for key, name, row, col in latency_metrics:
            if key in self.metrics_data and self.metrics_data[key]:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=self.metrics_data[key],
                        mode="lines+markers",
                        name=name,
                        marker=dict(size=4)
                    ),
                    row=row,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            title_text="vLLM Benchmark Results",
            showlegend=True,
            height=1200,
            width=1400,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Update y-axis labels and format
        fig.update_yaxes(title_text="Tokens/s", row=1, col=1, tickformat=".1f")
        fig.update_yaxes(title_text="Count", row=1, col=2, tickformat=".1f")
        fig.update_yaxes(title_text="Count", row=2, col=1, tickformat=".1f")
        fig.update_yaxes(title_text="Requests/s", row=2, col=2, tickformat=".1f")
        fig.update_yaxes(title_text="Count", row=3, col=1, tickformat=".1f")
        fig.update_yaxes(title_text="Count", row=3, col=2, tickformat=".1f")
        fig.update_yaxes(title_text="Seconds", row=4, col=1, tickformat=".1f")

        # Update x-axis format for all subplots
        for i in range(1, 5):
            for j in range(1, 3):
                fig.update_xaxes(tickformat=".1f", row=i, col=j)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"vllm_benchmark_{timestamp}.html")
        fig.write_html(output_file)
        logger.info(f"Benchmark results saved to {output_file}")

    async def run(self) -> None:
        """Run the benchmark."""
        self.start_time = time.time()

        async with aiohttp.ClientSession() as metrics_session, aiohttp.ClientSession() as load_session:
            self.metrics_session = metrics_session
            self.load_session = load_session
            # Create tasks for monitoring metrics and generating load
            monitor_task = asyncio.create_task(self.monitor_metrics())
            load_task = asyncio.create_task(self.generate_load())

            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [monitor_task, load_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the remaining task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.calculate_metrics()
        self.render_diagrams()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="vLLM Benchmark Tool")
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--metrics-url",
        default="http://localhost:8000/metrics",
        help="vLLM metrics endpoint URL (default: http://localhost:8000/metrics)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--qps", type=float, default=10, help="Target queries per second (default: 10)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--api-key", default=None, help="API key for vLLM authentication (optional)"
    )
    parser.add_argument(
        "--model",
        default="phala/deepseek-r1-70b",
        help="Model name for requests (default: phala/deepseek-r1-70b)",
    )
    parser.add_argument(
        "--payload-config",
        type=str,
        default='{"messages": [{"role": "user", "content": "Tell me about AI."}], "max_tokens": 100}',
        help='JSON string for request payload configuration (default: {"messages": [{"role": "user", "content": "Tell me about AI."}], "max_tokens": 100})',
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    try:
        payload_config = json.loads(args.payload_config)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid payload-config JSON: {e}")
        return

    benchmark = VLLMBenchmark(
        vllm_url=args.vllm_url,
        metrics_url=args.metrics_url,
        duration=args.duration,
        target_qps=args.qps,
        concurrency=args.concurrency,
        api_key=args.api_key,
        model=args.model,
        payload_config=payload_config,
    )

    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
