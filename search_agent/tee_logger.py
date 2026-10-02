"""
Tee logger infrastructure for Autonomous Research Agent.
Captures all stdout and stderr output to a timestamped file in evaluation_logs/
while preserving live terminal display and supporting verbose/clean modes.
"""

import datetime
import os
import sys

_ACTIVE_LOGGER = None


class _StreamTee:
    """Wraps a terminal stream and writes all output to a log file, and conditionally to terminal."""

    def __init__(self, terminal_stream, file_stream, verbose=True):
        self.terminal = terminal_stream
        self.file = file_stream
        self.verbose = verbose

    def write(self, data):
        # Always write to log file
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

        # Write to terminal if verbose is True
        if self.verbose:
            try:
                self.terminal.write(data)
                self.terminal.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass
        if self.verbose:
            try:
                self.terminal.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: 1)()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class TeeLogger:
    """Manages teeing stdout and stderr to a timestamped log file in UTF-8."""

    def __init__(self, log_dir="evaluation_logs", verbose=True):
        global _ACTIVE_LOGGER
        self.log_dir = log_dir
        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"research_run_{timestamp}.txt")
        self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1, errors="replace")

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

        self.stdout_tee = _StreamTee(self.orig_stdout, self.log_file, verbose=self.verbose)
        self.stderr_tee = _StreamTee(self.orig_stderr, self.log_file, verbose=self.verbose)

        sys.stdout = self.stdout_tee
        sys.stderr = self.stderr_tee
        _ACTIVE_LOGGER = self

        self.log_file.write(f"[Logging] Output is being recorded to: {self.log_path}\n")
        if self.verbose:
            try:
                self.orig_stdout.write(f"[Logging] Output is being recorded to: {self.log_path}\n")
                self.orig_stdout.flush()
            except Exception:
                pass

    @classmethod
    def get_current(cls):
        return _ACTIVE_LOGGER

    def get_log_path(self) -> str:
        return self.log_path

    def close(self):
        """Restore original stdout and stderr and close the log file."""
        global _ACTIVE_LOGGER
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.write(f"\n[Logging] Full execution log saved to: {self.log_path}\n")
            if self.verbose:
                try:
                    self.orig_stdout.write(f"\n[Logging] Full execution log saved to: {self.log_path}\n")
                    self.orig_stdout.flush()
                except Exception:
                    pass
            sys.stdout = self.orig_stdout
            sys.stderr = self.orig_stderr
            self.log_file.close()
        _ACTIVE_LOGGER = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
