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
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prometheus_client.parser import text_string_to_metric_families

from prompt import LONG_PROMPT_PAIRS, SHORT_PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        long_context: bool,
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
        self.long_context = long_context
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
            # New metrics for first token times and real-time token generation
            "first_token_times": [],  # List of first token times for each request
            "realtime_tokens_per_second": [],  # Real-time tokens generated per second
        }
        self.request_count = 0
        self.finished_request_count = 0  # Counter for finished requests
        self.request_success_count = 0  # Counter for successful requests
        self.metrics_session: Optional[aiohttp.ClientSession] = None
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        # For rate calculations
        self.prev_prompt_tokens = 0.0
        self.prev_generation_tokens = 0.0
        self.prev_time: Optional[float] = None
        self.stop_event = asyncio.Event()
        self.last_chunk_time: Optional[float] = None

        # For tracking tokens across all requests
        self.stream_tokens_count: int = 0
        self.stream_tokens_lock: asyncio.Lock = asyncio.Lock()

        self.generate_load_done = False

        # For recording requests
        self.records_file = open(
            os.path.join(self.output_dir, f"records_{get_timestamp()}.csv"),
            "w",
            encoding="utf-8",
        )
        self.records_file.write(
            "request_id,request_cost,prefill_cost,decode_cost,prompt_tokens,generation_tokens,request_retry_count,429_count,400_count,request_error_count\n"
        )

    async def fetch_metrics(self) -> None:
        """Fetch and parse metrics from vLLM /metrics endpoint."""
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
        try:
            async with self.metrics_session.get(self.metrics_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch metrics: HTTP {response.status}")
                    return
                metrics_text = await response.text()
                timestamp = time.time()

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

                # Update request success metric
                if metrics["request_success"] is not None:
                    # Use our local counter instead of the value from metrics API
                    self.metrics_data["request_success"].append(
                        self.request_success_count
                    )
                else:
                    self.metrics_data["request_success"].append(0)

                # Update finished request count from client-side counter
                self.metrics_data["finished_requests"].append(
                    self.finished_request_count
                )

                # Update time to first token metric
                if metrics["time_to_first_token"] is not None:
                    self.metrics_data["time_to_first_token"].append(
                        metrics["time_to_first_token"]
                    )
                else:
                    self.metrics_data["time_to_first_token"].append(0)

                # Update prefill time metric
                if metrics["prefill_time"] is not None:
                    self.metrics_data["prefill_time"].append(metrics["prefill_time"])
                else:
                    self.metrics_data["prefill_time"].append(0)

                # Update other latency metrics
                for metric_name in [
                    "e2e_latency",
                    "queue_time",
                    "inference_time",
                    "decode_time",
                ]:
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

        except Exception as e:
            self.metrics_data["time"].append(time.time())
            logger.error(f"Error fetching metrics: {e}")

    async def send_request(self, request_id: int) -> None:
        """Send a single request to vLLM API with optional API key authentication."""
        # Randomly choose content
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
            "stream": True,  # Enable streaming by default
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # Metrics
        _start_time = time.time()
        _request_start_time = 0
        _request_finish_time = 0
        _prefill_finish_time = 0
        _429_count = 0
        _400_count = 0
        _request_error_count = 0
        _request_retry_count = 0
        _prompt_tokens_count = 0
        _generation_tokens_count = 0

        while True:
            try:
                # Record request start time
                request_start_time = time.time()
                first_chunk_received = False

                _request_start_time = time.time()
                # Create a new session for each request
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.vllm_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=3 * 60 * 60,  # 3 hours timeout
                    ) as response:
                        if response.status == 200:
                            # Process streaming response
                            async for line in response.content:
                                if line:
                                    try:
                                        # Skip the "data: " prefix and parse JSON
                                        line_text = line.decode("utf-8").strip()
                                        if not line_text.startswith("data: "):
                                            continue

                                        json_str = line_text[
                                            6:
                                        ]  # Remove "data: " prefix
                                        if json_str == "[DONE]":
                                            _request_finish_time = time.time()
                                            _generation_tokens_count = (
                                                _prompt_tokens_count
                                                + _generation_tokens_count
                                            )
                                            break  # End of stream

                                        chunk = json.loads(json_str)
                                        # Calculate timing metrics from first chunk
                                        if not first_chunk_received:
                                            first_chunk_time = time.time()
                                            prefill_time = (
                                                first_chunk_time - request_start_time
                                            )

                                            # Store the prefill time
                                            self.metrics_data["prefill_time"].append(
                                                prefill_time
                                            )

                                            # Store the first token time
                                            self.metrics_data[
                                                "first_token_times"
                                            ].append(first_chunk_time - self.start_time)

                                            first_chunk_received = True
                                            _prefill_finish_time = first_chunk_time

                                            # For decode time, we'll use the time between first chunk and second chunk
                                            # This will be updated when we receive the second chunk
                                            self.last_chunk_time = first_chunk_time
                                        else:
                                            # Calculate decode time from second chunk onwards
                                            current_chunk_time = time.time()
                                            decode_time = (
                                                current_chunk_time
                                                - self.last_chunk_time
                                            )

                                            # Store the decode time
                                            self.metrics_data["decode_time"].append(
                                                decode_time
                                            )

                                            self.last_chunk_time = current_chunk_time

                                        # Parse usage
                                        if "usage" in chunk and chunk["usage"]:
                                            _prompt_tokens_count = (
                                                chunk["usage"].get("prompt_tokens", 0)
                                                or _prompt_tokens_count
                                            )
                                            _generation_tokens_count = (
                                                chunk["usage"].get(
                                                    "completion_tokens", 0
                                                )
                                                or _generation_tokens_count
                                            )

                                        # Track tokens received and calculate real-time tokens per second
                                        if (
                                            "choices" in chunk
                                            and len(chunk["choices"]) > 0
                                        ):
                                            if "delta" in chunk["choices"][0] and (
                                                "content"
                                                in chunk["choices"][0]["delta"]
                                                or "reasoning_content"
                                                in chunk["choices"][0]["delta"]
                                            ):
                                                content = chunk["choices"][0][
                                                    "delta"
                                                ].get(
                                                    "content",
                                                    chunk["choices"][0]["delta"].get(
                                                        "reasoning_content"
                                                    ),
                                                )
                                                if content:
                                                    # Approximate token count (rough estimate)
                                                    new_tokens = len(content.split())
                                                    async with self.stream_tokens_lock:
                                                        self.stream_tokens_count += (
                                                            new_tokens
                                                        )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse JSON from stream: {line_text}"
                                        )
                                    except Exception as e:
                                        logger.warning(traceback.format_exc())
                                        logger.warning(
                                            f"Error processing stream chunk: {e}"
                                        )

                            self.request_count += 1
                            self.finished_request_count += 1
                            self.request_success_count += 1
                            break
                        if response.status == 429:
                            _429_count += 1
                        elif response.status == 400:
                            _400_count += 1
                        elif response.status == 401:
                            logger.error(
                                "Authentication failed: Invalid or missing API key"
                            )
                            break
                        else:
                            _request_error_count += 1
                            logger.error(
                                f"Request failed: HTTP {response.status}, {await response.text()} at {time.time() - _start_time:.2f} seconds"
                            )
            except Exception as e:
                # Log detailed exception information
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(
                    f"Request error: {error_type}: {error_msg}, at {(time.time() - _start_time):.2f} seconds"
                )
                logger.error(traceback.format_exc())
                _request_error_count += 1

            # If we've reached here, the request failed and we should retry
            await asyncio.sleep(random.randint(0, 10))

        # Write request metrics to records
        # request_id, request_cost, prefill_cost, decode_cost, prompt_tokens, generation_tokens, request_retry_count, 429_count, 400_count, request_error_count
        self.records_file.write(
            f"{request_id},{int(_request_finish_time - _request_start_time)},{int(_prefill_finish_time - _request_start_time)},{int(_request_finish_time - _prefill_finish_time)},{_prompt_tokens_count},{_generation_tokens_count},{_request_retry_count},{_429_count},{_400_count},{_request_error_count}\n"
        )

    def _should_stop_early(self, check_tasks: bool = False) -> bool:
        """Check if the benchmark should stop early because all requests are completed or finished requests exceed concurrency.

        Args:
            check_tasks: If True, also check if there are no pending tasks.

        Returns:
            bool: True if the benchmark should stop early, False otherwise.
        """
        # Only check after 60 seconds to allow metrics to stabilize
        if time.time() - self.start_time <= 60:
            return False

        # Stop if finished requests exceed concurrency limit
        if self.finished_request_count >= self.concurrency:
            if self.finish_time is None:
                self.finish_time = time.time()
                logger.info(
                    f"Finished requests ({self.finished_request_count}) reached or exceeded concurrency limit ({self.concurrency}) at {self.finish_time:.2f} seconds"
                )
                logger.info(
                    f"Total time to complete all requests: {self.finish_time - self.start_time:.2f} seconds"
                )
            return True

        # Existing conditions for early stopping
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
        while time.time() - self.start_time < self.duration:
            start_time = time.time()
            await self.fetch_metrics()
            time_cost = time.time() - start_time
            if time_cost > 1:
                logger.warning(f"Metrics fetched in {time_cost:.2f} seconds")
            else:
                await asyncio.sleep(1)

            if self.generate_load_done:
                break

    async def handle_stream_tokens(self) -> None:
        """Handle stream tokens during the benchmark."""
        async with self.stream_tokens_lock:
            self.metrics_data["realtime_tokens_per_second"].append(
                self.stream_tokens_count
            )
            self.stream_tokens_count = 0

    async def scheduled_tasks(self) -> None:
        """Periodically run scheduled tasks during the benchmark."""
        _log_time = time.time()
        while time.time() - self.start_time < self.duration:
            if time.time() - _log_time > 10:
                _log_time = time.time()
                logger.info(f"Stream tokens count: {self.stream_tokens_count}")
                logger.info(f"{len(self.tasks)} requests has been sent")
                logger.info(f"Finished requests: {self.finished_request_count}")

            await self.handle_stream_tokens()

            await asyncio.sleep(1)

            # Stop if self.generate_load() is done
            if self.generate_load_done:
                break

    async def generate_load(self) -> None:
        """Generate load at target QPS with concurrency control."""
        interval = 1.0 / self.target_qps if self.target_qps > 0 else 0
        end_time = self.start_time + self.duration
        self.tasks = []

        logger.info("Generating load...")
        _request_id = 0
        while time.time() < end_time and not self.stop_event.is_set():
            if self._should_stop_early(check_tasks=True):
                logger.info(
                    "All requests have been completed. Stopping load generation early."
                )
                self.stop_event.set()
                break

            if len(self.tasks) < self.concurrency:
                self.tasks.append(asyncio.create_task(self.send_request(_request_id)))
                _request_id += 1
                if interval > 0:
                    await asyncio.sleep(interval)
            else:
                done, pending = await asyncio.wait(
                    self.tasks, timeout=interval, return_when=asyncio.FIRST_COMPLETED
                )
                self.tasks = list(pending) + [t for t in done if not t.exception()]

        if self.tasks:
            logger.warning(f"Waiting for {len(self.tasks)} tasks to complete")
            await asyncio.wait(self.tasks)

        logger.info("Cooling down...")
        await asyncio.sleep(10)  # wait for 10 seconds to cool down
        self.generate_load_done = True

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
            # Use our local counter instead of the value from metrics_data
            total_requests = self.request_success_count
            qps = total_requests / elapsed if elapsed > 0 else 0
        else:
            # Fallback to request_count if request_success is not available
            qps = self.request_count / elapsed if elapsed > 0 else 0

        # Calculate average QPS for each time point
        for i in range(len(times)):
            self.metrics_data["qps"].append(qps)
            # Make sure we don't access an index that doesn't exist
            if i < len(self.metrics_data["running_requests"]):
                self.metrics_data["concurrent_requests"].append(
                    self.metrics_data["running_requests"][i]
                )
            else:
                # If the index doesn't exist, use the last available value or 0
                last_value = (
                    self.metrics_data["running_requests"][-1]
                    if self.metrics_data["running_requests"]
                    else 0
                )
                self.metrics_data["concurrent_requests"].append(last_value)

        # Log summary metrics
        logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
        logger.info(f"Average QPS: {qps:.2f}")
        logger.info(f"Total successful requests: {self.request_success_count}")

    def render_diagrams(self) -> None:
        """Render metrics as an HTML page with Plotly diagrams."""
        times = np.array(self.metrics_data["time"])
        if not times.size:
            logger.warning("No data to render")
            return
        times = times - times[0]

        # Create a more comprehensive visualization with additional metrics
        fig = make_subplots(
            rows=5,  # Increased from 4 to 5 rows
            cols=2,
            subplot_titles=(
                "Token Processing Rate",
                "Running Requests",
                "Pending Requests",
                "QPS",
                "Concurrent Requests/s",
                "Request Success Count & Finished Requests",
                "Latency Metrics",
                "First Token Times",
                "Real-time Tokens per Second",
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
                marker=dict(size=2),
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
                marker=dict(size=2),
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
                    x=times,
                    y=self.metrics_data[key],
                    mode="lines+markers",
                    name=title,
                    marker=dict(size=2),
                ),
                row=row,
                col=col,
            )

        # Add request success count and finished requests in the same subplot
        if (
            "request_success" in self.metrics_data
            and self.metrics_data["request_success"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.metrics_data["request_success"],
                    mode="lines+markers",
                    name="Request Success Count",
                    marker=dict(size=2),
                    line=dict(color="green"),
                ),
                row=3,
                col=2,
            )

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
                    marker=dict(size=2),
                    line=dict(color="blue"),
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
                        marker=dict(size=2),
                    ),
                    row=row,
                    col=col,
                )

        # Add first token times scatter plot
        if (
            "first_token_times" in self.metrics_data
            and self.metrics_data["first_token_times"]
        ):
            # Create x-axis values (request indices)
            request_indices = list(range(len(self.metrics_data["first_token_times"])))

            fig.add_trace(
                go.Scatter(
                    x=request_indices,
                    y=self.metrics_data["first_token_times"],
                    mode="lines+markers",
                    name="First Token Times",
                    marker=dict(size=2, color="red"),
                ),
                row=4,
                col=2,
            )

        # Add real-time tokens per second
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.metrics_data["realtime_tokens_per_second"],
                mode="lines+markers",
                name="Real-time Tokens/s",
                marker=dict(size=2),
                line=dict(color="purple"),
            ),
            row=5,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title_text="vLLM Benchmark Results",
            showlegend=True,
            height=1500,  # Increased height to accommodate new plots
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
        fig.update_yaxes(title_text="Seconds", row=4, col=2, tickformat=".1f")
        fig.update_yaxes(title_text="Tokens/s", row=5, col=1, tickformat=".1f")

        # Update x-axis format for all subplots
        for i in range(1, 6):  # Updated to include the new row
            for j in range(1, 3):
                fig.update_xaxes(tickformat=".1f", row=i, col=j)

        output_file = os.path.join(
            self.output_dir, f"vllm_benchmark_{get_timestamp()}.html"
        )
        fig.write_html(output_file)
        logger.info(f"Benchmark results saved to {output_file}")

    async def run(self) -> None:
        """Run the benchmark."""
        self.start_time = time.time()

        async with aiohttp.ClientSession() as metrics_session:
            self.metrics_session = metrics_session
            # Create tasks for monitoring metrics and generating load
            monitor_task = asyncio.create_task(self.monitor_metrics())
            load_task = asyncio.create_task(self.generate_load())
            scheduled_task = asyncio.create_task(self.scheduled_tasks())
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [monitor_task, load_task, scheduled_task],
                return_when=asyncio.FIRST_COMPLETED,
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
        "--long-context",
        action="store_true",
        help="Use long context (default: False)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    benchmark = VLLMBenchmark(
        vllm_url=args.vllm_url,
        metrics_url=args.metrics_url,
        duration=args.duration,
        target_qps=args.qps,
        concurrency=args.concurrency,
        api_key=args.api_key,
        model=args.model,
        long_context=args.long_context,
    )

    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
