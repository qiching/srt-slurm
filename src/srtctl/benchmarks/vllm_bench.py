# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM benchmark runner using vllm bench serve."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("vllm-bench")
class VllmBenchRunner(BenchmarkRunner):
    """vLLM benchmark runner.

    Uses vllm bench serve to generate traffic. Supports profiling when
    profiling.type is set to "torch" or "nsys".

    Required config fields (in benchmark section):
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "128x256")

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
        - benchmark.num_prompts: Number of prompts per concurrency level (default: concurrency*10)
    """

    @property
    def name(self) -> str:
        return "vLLM-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/vllm-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "vllm-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for vllm-bench")
        elif b.isl <= 0:
            errors.append("benchmark.isl must be a positive integer for vllm-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for vllm-bench")
        elif b.osl <= 0:
            errors.append("benchmark.osl must be a positive integer for vllm-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for vllm-bench")
        else:
            try:
                concurrency_list = b.get_concurrency_list()
            except Exception:
                concurrency_list = []
                errors.append(
                    "benchmark.concurrencies must be a list of ints or an 'x'-separated string for vllm-bench"
                )

            if not concurrency_list:
                errors.append("benchmark.concurrencies must not be empty for vllm-bench")
            elif any(c <= 0 for c in concurrency_list):
                errors.append("benchmark.concurrencies values must be positive integers for vllm-bench")

        if isinstance(b.req_rate, int) and b.req_rate <= 0:
            errors.append("benchmark.req_rate must be a positive integer or 'inf' for vllm-bench")
        if isinstance(b.req_rate, str) and b.req_rate.strip() in ("0", "0.0"):
            errors.append("benchmark.req_rate must be a positive number or 'inf' for vllm-bench")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrencies),
            str(b.req_rate) if b.req_rate is not None else "inf",
            str(b.num_prompts) if b.num_prompts is not None else "0",
        ]
