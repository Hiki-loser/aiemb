"""llama-completion query engine. Uses Popen with manual stdin close."""

import os
import subprocess
import logging
import time

import config


class LLMEngine:
    """Manages llama-completion queries via Popen + stdin.
    First query cold-loads model (~3s), subsequent queries use page cache (~1-2s)."""

    def __init__(self):
        self._log = logging.getLogger("pipeline")

    def query(self, prompt):
        """Run one-shot inference. Returns generated text."""
        cmd = [
            config.LLAMA_LD,
            config.LLAMA_BIN,
            "-m", config.LLAMA_MODEL,
            "-f", "/dev/stdin",
            "--temp", str(config.LLM_TEMP),
            "-n", str(config.LLM_TOKENS),
            "-t", str(config.LLM_THREADS),
            "--cpu-mask", config.LLM_CPU_MASK,
            "--no-conversation",
            "--no-display-prompt",
            "--single-turn",
        ]

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = config.LLAMA_LD_PATH

        self._log.debug("LLM query: %d chars prompt", len(prompt))

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=config.LLAMA_WORKDIR,
                env=env,
            )

            # Write prompt and close stdin to signal EOF
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()

            # Wait for process with timeout
            deadline = time.time() + 30
            output = b""
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    chunk = proc.stdout.read(4096)
                    if chunk:
                        output += chunk
                    else:
                        break
                except Exception:
                    break
                time.sleep(0.05)

            if proc.poll() is None:
                proc.kill()
                self._log.error("LLM timeout after 30s")
                return ""

            stdout = output.decode(errors="replace").strip()

        except Exception as e:
            self._log.error("LLM error: %s", e)
            return ""

        # Extract the generated DEVICE|ACTION|PARAMS line
        for line in stdout.split("\n"):
            line = line.strip()
            if line.upper().startswith("DEVICE:"):
                return line

        lines = [l.strip() for l in stdout.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    def shutdown(self):
        """No-op."""
        pass
