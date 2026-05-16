"""Per-channel online inference engine for the FastAPI serving layer.

ChannelInferenceEngine wraps a trained TelemanomLSTM and replicates the
offline anomaly-scoring pipeline (model/scoring.py) in an O(1)-per-tick
stateful fashion, so each call to .step() produces exactly what the offline
pipeline would have produced for that timestep.

Design notes
------------
- ``ScoringParams`` is imported from ``model.io`` — defined there to keep the
  model layer self-contained; re-exported here for callers that only import api.
- ``TelemetryEvent`` is defined here (Step 4).  It will move to ``api/models.py``
  in Step 5 and this module will import it from there.
- The engine requires PyTorch at import time.  Use ``pytest.importorskip("torch")``
  in tests.
- The model must already be in ``.eval()`` mode before being passed to the
  constructor; the lifespan handler (``api/app.py``) is responsible for that.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch

from spacecraft_telemetry.api.models import TelemetryEvent
from spacecraft_telemetry.model.io import ScoringParams

if TYPE_CHECKING:
    from spacecraft_telemetry.model.architecture import TelemanomLSTM

# Re-export so callers can ``from spacecraft_telemetry.api.inference import ScoringParams``.
__all__ = ["ChannelInferenceEngine", "ScoringParams", "TelemetryEvent"]


# ---------------------------------------------------------------------------
# ChannelInferenceEngine
# ---------------------------------------------------------------------------


class ChannelInferenceEngine:
    """Stateful per-channel engine for online anomaly detection.

    Maintains rolling buffers for window accumulation, EWMA smoothing, and
    K-trailing anomaly detection.  The math is byte-identical to the offline
    pipeline (``model/scoring.py``) past warmup:

    - EWMA: ``alpha * |e_t| + (1 - alpha) * s_{t-1}`` with ``adjust=False``
      (matches ``pd.Series.ewm(span=..., adjust=False)``).
    - Threshold: ``mean(prior_Tw) + z * std(prior_Tw, ddof=0)`` where
      ``prior_Tw`` is the ``threshold_window`` smoothed errors *before* the
      current tick (mirrors ``.shift(1)`` in ``dynamic_threshold``).
      Returns ``math.inf`` until ``threshold_window`` prior values are
      accumulated (conservative warmup; first threshold is computed at tick
      ``window_size + threshold_window``).
    - Anomaly flag: ``True`` iff the last ``threshold_min_anomaly_len`` raw
      flags (including current) are all ``True`` — trailing-edge variant of
      ``flag_anomalies`` with a ``K-1`` leading-edge lag.

    Warmup summary
    --------------
    - First ``window_size`` ticks: ``prediction=None``, ``threshold=None``.
    - Next ``threshold_window`` ticks: ``threshold=None`` (smoothed error
      is non-None once model warmup ends).
    - From tick ``window_size + threshold_window`` onwards: all fields
      non-None; values match offline pipeline to floating-point precision.
    """

    def __init__(
        self,
        *,
        mission: str,
        channel: str,
        model: TelemanomLSTM,
        window_size: int,
        params: ScoringParams,
        device: torch.device,
    ) -> None:
        self._mission = mission
        self._channel = channel
        self._model = model
        self._window_size = window_size
        self._params = params
        self._device = device

        # Rolling input buffer — accumulates window_size ticks before inference.
        self._window_buf: deque[float] = deque(maxlen=window_size)

        # Pre-allocated input tensor — reused every tick to avoid per-tick
        # allocation + reshape. Filled in-place from _window_buf each step.
        self._input_buf: torch.Tensor = torch.zeros(
            1, window_size, 1, dtype=torch.float32, device=device
        )

        # EWMA smoothing state (scalar, updated each step post-warmup).
        self._alpha: float = 2.0 / (params.error_smoothing_window + 1)
        self._s_prev: float | None = None

        # Smoothed error ring buffer for threshold computation.
        # Size = Tw + 1 so "prior" (all entries except the current tick) holds
        # exactly Tw values when the buffer is full.  Pre-allocated to avoid
        # O(Tw) list-copy + numpy allocation per tick.
        _tw = params.threshold_window
        self._smoothed_np: np.ndarray[np.float64, np.dtype[np.float64]] = (
            np.empty(_tw + 1, dtype=np.float64)
        )
        self._smoothed_pos: int = 0   # next write position
        self._smoothed_count: int = 0  # total values written
        self._prior_buf: np.ndarray[np.float64, np.dtype[np.float64]] = (
            np.empty(_tw, dtype=np.float64)
        )

        # K-trailing anomaly flag buffer.
        self._raw_flag_buf: deque[bool] = deque(
            maxlen=params.threshold_min_anomaly_len
        )

    @torch.no_grad()
    def step(
        self, value: float, timestamp: datetime, is_anomaly: bool
    ) -> TelemetryEvent:
        """Process one tick and return a TelemetryEvent.

        Args:
            value:          Normalised channel value for this tick.
            timestamp:      Timestamp of this tick (used verbatim in the event).
            is_anomaly: Ground-truth anomaly label from the Parquet file.

        Returns:
            TelemetryEvent with all fields populated.  Fields that cannot yet be
            computed (warmup) are set to ``None``; ``is_anomaly_predicted`` is
            always a bool (``False`` during warmup).
        """
        # 1. Accumulate input buffer.
        self._window_buf.append(value)

        # 2. Model warmup — not enough ticks to fill a window yet.
        if len(self._window_buf) < self._window_size:
            return TelemetryEvent(
                timestamp=timestamp,
                mission=self._mission,
                channel=self._channel,
                value_normalized=value,
                prediction=None,
                residual=None,
                smoothed_error=None,
                threshold=None,
                is_anomaly_predicted=False,
                is_anomaly=is_anomaly,
            )

        # 3. Predict (model is already in eval() mode; @torch.no_grad() wraps step).
        self._input_buf[0, :, 0] = torch.tensor(
            list(self._window_buf), dtype=torch.float32
        )
        prediction: float = self._model(self._input_buf).item()

        residual = value - prediction
        e_abs = abs(residual)

        # 4. EWMA smoothing: s_t = alpha * |e_t| + (1 - alpha) * s_{t-1}.
        #    s_0 = |e_0| (adjust=False initialisation).
        if self._s_prev is None:
            s_t = e_abs
        else:
            s_t = self._alpha * e_abs + (1.0 - self._alpha) * self._s_prev
        self._s_prev = s_t

        # 5. Threshold: prior = last Tw smoothed values *before* this tick.
        #    Write s_t to ring buffer; prior = all slots except the one just written.
        _tw1 = self._params.threshold_window + 1
        _tw = self._params.threshold_window
        self._smoothed_np[self._smoothed_pos] = s_t
        self._smoothed_pos = (self._smoothed_pos + 1) % _tw1
        self._smoothed_count += 1

        if self._smoothed_count <= _tw:
            threshold_val: float = math.inf
        else:
            # Buffer is full: prior = Tw elements starting from _smoothed_pos
            # (the oldest element, which wraps around to the current write head).
            start = self._smoothed_pos
            end = (start + _tw) % _tw1
            if start < end:
                self._prior_buf[:] = self._smoothed_np[start:end]
            else:
                first = _tw1 - start
                self._prior_buf[:first] = self._smoothed_np[start:]
                self._prior_buf[first:] = self._smoothed_np[:end]
            threshold_val = float(
                self._prior_buf.mean()
                + self._params.threshold_z * self._prior_buf.std(ddof=0)
            )

        # 6. K-trailing anomaly flag.
        raw_t = s_t > threshold_val
        self._raw_flag_buf.append(raw_t)
        K = self._params.threshold_min_anomaly_len
        is_anomaly_predicted = len(self._raw_flag_buf) == K and all(
            self._raw_flag_buf
        )

        return TelemetryEvent(
            timestamp=timestamp,
            mission=self._mission,
            channel=self._channel,
            value_normalized=value,
            prediction=float(prediction),
            residual=float(residual),
            smoothed_error=float(s_t),
            threshold=None if math.isinf(threshold_val) else float(threshold_val),
            is_anomaly_predicted=is_anomaly_predicted,
            is_anomaly=is_anomaly,
        )
