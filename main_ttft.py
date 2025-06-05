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
from pydantic import BaseModel

from prompt import LONG_PROMPT_PAIRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TIMEOUT = 60 * 10
WARMUP_TIME = 6  # 2 minutes
# PROMPT_LENGTH = 9000  # 4500 + 500 = 5000
# MAX_TOKENS = 9000 + 2000
PROMPT_LENGTH = 4500
MAX_TOKENS = 4500 + 500


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RequestContext(BaseModel):
    """Context for tracking request state and metrics."""

    payload: dict = None
    headers: dict = None
    request_start_time: float
    first_token_received: bool = False
    first_token_time: Optional[float] = None
    connected_time: Optional[float] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    status_code: Optional[int] = None
    should_break: bool = False
    should_continue: bool = False


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
        self.initial_max_connections = initial_max_connections
        self.max_connections = initial_max_connections
        self.duration = duration
        self.target_qps = target_qps
        self.current_qps = target_qps * 1  # Start with 10x QPS
        self.api_key = api_key
        self.model = model

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
        self.failed_requests = 0
        self.start_time: Optional[float] = None
        self.stop_event = asyncio.Event()

        # TTFT tracking for gradual adjustments
        self.adjustment_step = 30
        self.timeout_count = 0
        self.last_timeout_check = 0
        self.timeout_check_interval = 30

        # Output directory
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_payload(self, ctx: RequestContext) -> None:
        """Prepare payload and headers for the request."""
        prompt_pair = random.choice(LONG_PROMPT_PAIRS)
        content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
        if len(content) < PROMPT_LENGTH:
            # Expand the prompt by repeating it until it reaches the desired length
            while len(content) < PROMPT_LENGTH:
                content += "\n\n" + content
            content = content[:PROMPT_LENGTH]  # Trim to exact length if needed

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        ctx.payload = payload
        ctx.headers = headers

    async def _process_stream_line(
        self,
        line: str,
        ctx: RequestContext,
    ) -> None:
        """Process a single line from the stream response."""
        line_text = line.decode("utf-8").strip()
        if not line_text.startswith("data: "):
            ctx.should_continue = True
            return

        json_str = line_text[6:]  # Remove "data: " prefix
        if json_str == "[DONE]":
            ctx.should_break = True
            return

        try:
            data = json.loads(json_str)
            if "usage" in data and not self.is_warmup():
                self._update_token_usage(data["usage"])

            if not ctx.first_token_received:
                ctx.first_token_time = time.time()
                ctx.first_token_received = True
                if not self.is_warmup():
                    ttft = ctx.first_token_time - ctx.request_start_time
                    self.metrics_data["ttft"].append(ttft)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from stream: {line_text}")
        except Exception as e:
            logger.warning(f"Error processing stream chunk: {e}")

    def _update_token_usage(self, usage: dict) -> None:
        """Update token usage statistics."""
        if "prompt_tokens" in usage:
            self.total_prompt_tokens += usage["prompt_tokens"]
        if "completion_tokens" in usage:
            self.total_generation_tokens += usage["completion_tokens"]

    async def is_ttft_timeout(self, ctx: RequestContext) -> bool:
        """Check if request has timeout waiting for first token."""
        if time.time() - ctx.connected_time > 20:
            logger.error(f"TTFT > 20, cancel request")
            return True
        return False

    def is_warmup(self) -> bool:
        return time.time() < self.start_time + WARMUP_TIME

    async def send_request(self) -> None:
        """Send a request to the vLLM service and process the response."""
        ctx = RequestContext(request_start_time=time.time())
        self.prepare_payload(ctx)
        self.active_requests += 1
        self.total_requests += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=ctx.payload,
                    headers=ctx.headers,
                    timeout=TIMEOUT,
                ) as response:
                    ctx.status_code = response.status
                    if response.status == 200:
                        ctx.connected_time = time.time()
                        async for line in response.content:
                            # Check for timeout
                            if not ctx.first_token_received:
                                if await self.is_ttft_timeout(ctx):
                                    self.timeout_count += 1
                                    self.failed_requests += 1
                                    return
                            if line:
                                await self._process_stream_line(line, ctx)
                            if ctx.should_break:
                                break
                            if ctx.should_continue:
                                ctx.should_continue = False
                                continue

                        self.successful_requests += 1
                    elif response.status == 429:
                        return
                    else:
                        logger.error(f"Request failed: HTTP {response.status}")
        except asyncio.TimeoutError:
            logger.warning(f"Request timed out after {TIMEOUT} seconds")
            self.timeout_count += 1
            self.failed_requests += 1
        except Exception as e:
            # Log detailed exception information
            logger.error(f"Request error: {str(e)}")
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Request error: {error_type}: {error_msg}")
            logger.error(traceback.format_exc())
        finally:
            self.active_requests -= 1

    async def adjust_max_connections(self, current_time: float) -> None:
        """Adjust max_connections based on timeout rate."""
        if self.active_requests > self.max_connections:
            return
        if current_time - self.last_timeout_check < self.timeout_check_interval:
            return

        if self.timeout_count > 0:
            self.max_connections = self.max_connections - self.adjustment_step
            logger.info(
                f"Reducing max_connections to {self.max_connections} due to high timeout rate"
            )
        else:
            if self.active_requests >= self.max_connections:
                self.max_connections = self.max_connections + self.adjustment_step
                logger.info(
                    f"Increasing max_connections to {self.max_connections} due to low timeout rate"
                )

        # Reset counters
        self.timeout_count = 0
        self.last_timeout_check = current_time

    async def monitor_metrics(self) -> None:
        """Periodically collect and update metrics."""
        while not self.stop_event.is_set():
            if time.time() < self.start_time + WARMUP_TIME:
                await asyncio.sleep(1)
                continue

            current_time = time.time()
            elapsed = current_time - self.start_time

            # Record metrics
            self.metrics_data["time"].append(elapsed)
            self.metrics_data["running_requests"].append(self.active_requests)
            self.metrics_data["max_connections"].append(self.max_connections)

            # Calculate QPS (over last 5 seconds)
            qps = 0
            if len(self.metrics_data["time"]) > 1:
                time_window = 5
                recent_requests = sum(
                    1 for t in self.metrics_data["time"] if t > elapsed - time_window
                )
                qps = recent_requests / time_window
                self.metrics_data["qps"].append(qps)

            # Check timeouts and adjust max_connections
            await self.adjust_max_connections(current_time)

            # Adjust QPS based on active connections
            if self.active_requests >= self.max_connections:
                self.current_qps = self.target_qps
            else:
                self.current_qps = self.target_qps * 10

            # Calculate token processing rates
            prompt_tokens_rate = 0
            generation_tokens_rate = 0
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
                    f"Sent: {self.total_requests}, "
                    f"Active: {self.active_requests}, "
                    f"Success: {self.successful_requests}, "
                    f"Failed: {self.failed_requests}, "
                    f"Max Conn: {self.max_connections}, "
                    f"Current QPS: {self.current_qps:.2f}, "
                    f"Avg TTFT: {avg_ttft:.2f}s, "
                    f"QPS: {qps:.2f}, "
                    f"Prompt tps: {prompt_tokens_rate:.2f}, "
                    f"Gen tps: {generation_tokens_rate:.2f}"
                )

            await asyncio.sleep(1)

    async def generate_load(self) -> None:
        """Generate load with dynamic concurrency control."""
        interval = 1.0 / self.current_qps if self.current_qps > 0 else 0
        end_time = self.start_time + self.duration

        logger.info(f"Warmup for {WARMUP_TIME} seconds")
        while time.time() < end_time and not self.stop_event.is_set():
            if self.active_requests < self.max_connections:
                asyncio.create_task(self.send_request())
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

        # Update y-axis labels and formatting
        fig.update_yaxes(title_text="Seconds", row=1, col=1, tickformat=".0f")
        fig.update_yaxes(title_text="Count", row=1, col=2, tickformat=".0f")
        fig.update_yaxes(title_text="Requests/s", row=2, col=1, tickformat=".0f")
        fig.update_yaxes(title_text="Count", row=2, col=2, tickformat=".0f")
        fig.update_yaxes(title_text="Tokens/s", row=3, col=1, tickformat=".0f")
        fig.update_yaxes(title_text="Tokens/s", row=3, col=2, tickformat=".0f")

        # Update x-axis formatting for all subplots
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(tickformat=".0f", row=i, col=j)

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
        logger.info(f"Avg Input TPS: {self.total_prompt_tokens / self.duration:.2f}")
        logger.info(
            f"Avg Output TPS: {self.total_generation_tokens / self.duration:.2f}"
        )
        logger.info(
            f"Avg TPS: {(self.total_prompt_tokens + self.total_generation_tokens) / self.duration:.2f}"
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
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="Use long context for requests (default: False)",
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
