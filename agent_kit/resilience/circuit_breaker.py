"""Thread-safe circuit breaker.

Three-state machine:
- CLOSED: all calls allowed; consecutive failures count up to threshold.
- OPEN:   calls rejected; transitions to HALF_OPEN after reset_timeout_s.
- HALF_OPEN: a probe window; ``success_threshold_in_half_open`` consecutive
  successes close the circuit, any single failure reopens it.

Designed for wrapping a Model so a flaky provider doesn't take down a fleet.
"""
from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        success_threshold_in_half_open: int = 2,
        *,
        clock=time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be > 0")
        if success_threshold_in_half_open < 1:
            raise ValueError("success_threshold_in_half_open must be >= 1")
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.success_threshold = success_threshold_in_half_open
        self._clock = clock
        self._lock = threading.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        # Take a snapshot in case the periodic transition flips us.
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def allow(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._half_open_successes = 0
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in probe window reopens.
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_successes = 0
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = self._clock()

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == CircuitState.OPEN:
            elapsed = self._clock() - self._opened_at
            if elapsed >= self.reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
