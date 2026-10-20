import io
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Total labeled brain volume below this threshold (mm³) is treated as a clear failure.
DEFAULT_MIN_BRAIN_VOLUME_MM3 = 10_000
DEFAULT_CONSOLE_LOG_FILENAME = "processing_log.txt"


def _apply_carriage_returns(text: str) -> str:
    """Emulate terminal carriage-return (\r) overwriting on a single line."""
    if "\r" not in text:
        return text
    line = ""
    for segment in text.split("\r"):
        if segment:
            line = segment + line[len(segment) :]
    return line


class _TeeStream:
    """Stream wrapper that forwards raw output to a terminal stream and clean lines to a log file."""

    def __init__(self, terminal_stream, log_sink):
        self._terminal = terminal_stream
        self._log_sink = log_sink
        self._line_buf = ""

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        written = self._terminal.write(data)
        self._terminal.flush()

        parts = data.split("\n")
        for part in parts[:-1]:
            line = _apply_carriage_returns(self._line_buf + part)
            self._log_sink.write_line(line + "\n")
            self._line_buf = ""
        self._line_buf = _apply_carriage_returns(self._line_buf + parts[-1])
        return written

    def flush(self) -> None:
        self._terminal.flush()
        self._log_sink.flush()

    def flush_pending(self) -> None:
        if self._line_buf:
            line = _apply_carriage_returns(self._line_buf)
            if line:
                self._log_sink.write_line(line + "\n")
            self._line_buf = ""

    def isatty(self) -> bool:
        return getattr(self._terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._terminal.fileno()

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._terminal, "errors", "replace")

    def __getattr__(self, name):
        return getattr(self._terminal, name)


class ConsoleTeeLogger:
    """Capture stdout and stderr to both the terminal and a text file."""

    def __init__(self):
        self._buffer = io.StringIO()
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._tee_stdout = None
        self._tee_stderr = None

    def write_line(self, text: str) -> None:
        if self._file is not None:
            self._file.write(text)
            self._file.flush()
        else:
            self._buffer.write(text)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def bind_output_folder(self, output_folder: str, filename: str = DEFAULT_CONSOLE_LOG_FILENAME) -> str:
        """Open the target .txt file in output_folder and flush any buffered early output."""
        os.makedirs(output_folder, exist_ok=True)
        log_path = os.path.join(output_folder, filename)
        self._file = open(log_path, "w", encoding="utf-8")
        buffered = self._buffer.getvalue()
        if buffered:
            self._file.write(buffered)
            self._file.flush()
        self._buffer = io.StringIO()
        return log_path

    def start(self) -> "ConsoleTeeLogger":
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_stdout = _TeeStream(self._orig_stdout, self)
        self._tee_stderr = _TeeStream(self._orig_stderr, self)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr
        return self

    def close(self) -> None:
        if self._tee_stdout is not None:
            self._tee_stdout.flush_pending()
        if self._tee_stderr is not None:
            self._tee_stderr.flush_pending()
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


@contextmanager
def capture_console_output():
    """Context manager that tees stdout and stderr into a ConsoleTeeLogger."""
    logger = ConsoleTeeLogger().start()
    try:
        yield logger
    finally:
        logger.close()


def total_brain_volume_mm3(parcellation: np.ndarray) -> int:
    """Return the total labeled brain volume in mm³ on the 1 mm isotropic grid."""
    return int(np.count_nonzero(parcellation))


class ProcessingLog:
    """Collect failed, skipped, and suspiciously low-volume cases for batch QC."""

    def __init__(self, min_brain_volume_mm3: int = DEFAULT_MIN_BRAIN_VOLUME_MM3):
        self.min_brain_volume_mm3 = min_brain_volume_mm3
        self.records = []

    def add_failed(self, input_path: str, case_id: str, error_message: str) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "failed",
                "reason": error_message,
                "total_brain_volume_mm3": "",
            }
        )

    def add_skipped(self, input_path: str, case_id: str, skip_reason: str) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "skipped",
                "reason": skip_reason,
                "total_brain_volume_mm3": "",
            }
        )

    def add_low_volume(self, input_path: str, case_id: str, total_volume: int) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "low_volume",
                "reason": (
                    f"Total brain volume ({total_volume} mm³) is below "
                    f"threshold ({self.min_brain_volume_mm3} mm³)."
                ),
                "total_brain_volume_mm3": total_volume,
            }
        )

    def check_low_volume(self, input_path: str, case_id: str, parcellation: np.ndarray) -> bool:
        total_volume = total_brain_volume_mm3(parcellation)
        if total_volume < self.min_brain_volume_mm3:
            self.add_low_volume(input_path, case_id, total_volume)
            return True
        return False

    def save(self, output_folder: str, filename: str = "failed_cases.csv") -> None:
        if not self.records:
            print("No failed, skipped, or low-volume cases to report.")
            return

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        pd.DataFrame(self.records).to_csv(output_path, index=False)
        print(f"Wrote {len(self.records)} issue(s) to {output_path}")
