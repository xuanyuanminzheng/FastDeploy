"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
Multi-stream shared expert overlap utility for MoE layers.

Runs the shared expert MLP on a separate CUDA stream concurrently with
the routed expert path on the main stream, hiding shared expert latency
during decode (small batch) scenarios.

Usage in model code:
    overlap = SharedExpertOverlap(shared_experts_layer)

    # In forward():
    overlap_started = overlap.maybe_start(x, forward_meta)
    routed_out = self.experts(x, self.gate, forward_meta)
    if overlap_started:
        shared_out = overlap.finish()
    else:
        shared_out = self.shared_experts(x)
    out = routed_out + shared_out
"""

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.platforms import current_platform

# Global singleton auxiliary CUDA stream (lazy-initialized, shared by all MoE layers)
_aux_stream = None
_aux_stream_initialized = False


def get_aux_stream():
    """Get or create the global auxiliary CUDA stream (singleton)."""
    global _aux_stream, _aux_stream_initialized
    if not _aux_stream_initialized:
        _aux_stream_initialized = True
        if current_platform.is_cuda():
            _aux_stream = paddle.device.cuda.Stream()
            logger.info("SharedExpertOverlap: created auxiliary CUDA stream for shared expert overlap")
    return _aux_stream


class SharedExpertOverlap:
    """
    Manages multi-stream overlap between routed experts (main stream)
    and shared experts (auxiliary stream).

    Designed to be instantiated once per MoE layer and reused across forward calls.

    Memory safety: Since Paddle lacks tensor.record_stream(), we hold a Python
    reference to the input tensor until the aux stream completes (guaranteed by
    the event-wait in finish()).
    """

    def __init__(self, shared_experts: nn.Layer):
        self._shared_experts = shared_experts
        self._stream = get_aux_stream()

        # Pre-allocate CUDA events (reused across calls to avoid allocation overhead)
        if self._stream is not None:
            self._start_event = paddle.device.cuda.Event()
            self._done_event = paddle.device.cuda.Event()
        else:
            self._start_event = None
            self._done_event = None

        # Per-call state
        self._input_ref = None
        self._output = None
        self._overlap_active = False

    @property
    def enabled(self) -> bool:
        """Whether overlap infrastructure is available."""
        return self._stream is not None and envs.FD_SHARED_EXPERT_OVERLAP and current_platform.is_cuda()

    def should_overlap(self, x: paddle.Tensor, forward_meta=None) -> bool:
        """
        Determine whether to overlap for the current forward call.

        Overlap is only beneficial for small batches (decode). For large
        batches (prefill), the GPU is already saturated.
        """
        if not self.enabled:
            return False

        # Skip when CUDA graph is active (multi-stream incompatible)
        if forward_meta is not None and getattr(forward_meta, "step_use_cudagraph", False):
            return False

        token_count = x.shape[0]
        threshold = envs.FD_SHARED_EXPERT_OVERLAP_THRESHOLD
        return token_count <= threshold

    def maybe_start(self, x: paddle.Tensor, forward_meta=None) -> bool:
        """
        Conditionally launch shared experts on the auxiliary stream.

        Returns True if overlap was started (caller should later call finish()),
        False if caller should run shared experts sequentially.

        Must be called BEFORE the routed expert computation begins.
        """
        if not self.should_overlap(x, forward_meta):
            return False

        # Hold reference to prevent GC while aux stream reads the tensor
        self._input_ref = x
        self._overlap_active = True

        main_stream = paddle.device.current_stream()

        # Record current point on main stream
        self._start_event.record(main_stream)

        # Launch shared experts on aux stream
        with paddle.device.stream_guard(self._stream):
            # Aux stream waits for main stream's prior ops to complete
            self._stream.wait_event(self._start_event)
            # Run shared expert MLP
            self._output = self._shared_experts(x)
            # Record completion
            self._done_event.record(self._stream)

        return True

    def finish(self) -> paddle.Tensor:
        """
        Wait for the auxiliary stream to complete and return the shared expert output.

        Must be called AFTER the routed expert computation finishes on the main stream.
        """
        if not self._overlap_active:
            raise RuntimeError("SharedExpertOverlap.finish() called without active overlap")

        # Main stream waits for aux stream to complete shared expert computation
        main_stream = paddle.device.current_stream()
        main_stream.wait_event(self._done_event)

        # Retrieve output and clear state
        output = self._output
        self._output = None
        self._input_ref = None
        self._overlap_active = False

        return output

    def run_sequential(self, x: paddle.Tensor) -> paddle.Tensor:
        """Fallback: run shared experts sequentially on main stream."""
        return self._shared_experts(x)
