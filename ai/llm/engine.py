"""
SPECTRA - Local LLM Engine

Real on-device text generation via ONNX Runtime GenAI.

Design notes
------------
The backend is `onnxruntime-genai` rather than transformers or llama.cpp
because it is the same runtime that exposes the Qualcomm QNN execution
provider. A model that runs here can move to the Hexagon NPU without any
change in the application layer, which is the entire point of SPECTRA.

Qwen3 INT4 is the default. 1.7B (1.35 GB) is the smallest size that answers
questions about its own hardware reliably; 0.6B (511 MB) stays in the catalog
for constrained machines.

API notes (verified against onnxruntime-genai 0.17.x)
-----------------------------------------------------
1. `Tokenizer.apply_chat_template(messages: str, *, template_str=None,
   tools=None, add_generation_prompt=True)` takes a **JSON string**, not a
   list of dicts. Passing a list raises TypeError, and a caller that swallows
   that exception silently falls back to a hand-rolled prompt.
2. The binding accepts no `enable_thinking` kwarg, but `chat_template.jinja`
   honours it. To suppress Qwen3 reasoning we render the template ourselves
   with `enable_thinking: false` passed through Jinja, or append a
   pre-closed empty think block.
3. `GeneratorParams` is configured via `set_search_options(**kwargs)`; prompt
   tokens go in through `Generator.append_tokens()`.
4. Search option `max_length` is the **total** sequence length (prompt +
   generated), not a generation budget.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("LLM")


# ── Model catalog ────────────────────────────────────────────────────────────

CATALOG: dict[str, dict] = {
    "qwen3_1_7b_int4": {
        "name": "Qwen3-1.7B (INT4, KLD block-128)",
        "repo": "onnx-community/Qwen3-1.7B-ONNX",
        "subfolder": "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128",
        "size_mb": 1355,
        "params": "1.7B",
        "quantization": "int4 (KLD, block 128)",
        "license": "Apache-2.0",
        "notes": (
            "Default SPECTRA assistant model. The 0.6B model answers "
            "questions about its own hardware incorrectly; 1.7B is the "
            "smallest size that is reliably competent."
        ),
    },
    "qwen3_0_6b_int4": {
        "name": "Qwen3-0.6B (INT4, KLD block-128)",
        "repo": "onnx-community/Qwen3-0.6B-ONNX",
        "subfolder": "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128",
        "size_mb": 511,
        "params": "0.6B",
        "quantization": "int4 (KLD, block 128)",
        "license": "Apache-2.0",
        "notes": (
            "Low-memory option. Too small to answer hardware questions "
            "reliably; keep as a fallback for constrained machines."
        ),
    },
    "qwen3_0_6b_int8": {
        "name": "Qwen3-0.6B (INT8)",
        "repo": "onnx-community/Qwen3-0.6B-ONNX",
        "subfolder": "onnxruntime/cpu_and_mobile/cpu-int8",
        "size_mb": 800,
        "params": "0.6B",
        "quantization": "int8",
        "license": "Apache-2.0",
        "notes": "Higher accuracy than int4, roughly 60% larger.",
    },
}

DEFAULT_MODEL = "qwen3_1_7b_int4"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Qwen3 opens reasoning with <think>\n\n and closes it with </think>.
_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


@dataclass
class GenerationResult:
    """One generation, with the telemetry a competition demo needs."""

    success: bool
    text: str = ""
    reasoning: str = ""
    prompt: str = ""
    model_id: str = ""
    model_name: str = ""
    device: str = "CPU"
    provider: str = "CPUExecutionProvider"
    backend: str = "onnxruntime-genai"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0          # time to first token
    total_ms: float = 0.0
    tokens_per_second: float = 0.0
    requested_device: Optional[str] = None
    fell_back_to_cpu: bool = False
    error_message: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "text": self.text,
            "reasoning": self.reasoning,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "device": self.device,
            "provider": self.provider,
            "backend": self.backend,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "requested_device": self.requested_device,
            "fell_back_to_cpu": self.fell_back_to_cpu,
            "error": self.error_message,
            "notes": self.notes,
        }

    def print_report(self) -> None:
        print(f"\n{'─' * 60}")
        print("  LOCAL LLM GENERATION")
        print(f"{'─' * 60}")
        print(f"  Model      : {self.model_name}")
        print(f"  Backend    : {self.backend}")
        print(f"  Device     : {self.device} ({self.provider})")
        if self.requested_device and self.requested_device != self.device:
            print(f"  Requested  : {self.requested_device} [FALLBACK]")
        print(f"  Tokens     : {self.prompt_tokens} in / {self.completion_tokens} out")
        print(f"  TTFT       : {self.ttft_ms:.1f} ms")
        print(f"  Total      : {self.total_ms:.1f} ms")
        print(f"  Throughput : {self.tokens_per_second:.2f} tok/s")
        if self.error_message:
            print(f"  ERROR      : {self.error_message}")
        else:
            print(f"\n  Response:\n  {self.text}")
        print()


@dataclass
class _LoadedLLM:
    model: object = None
    tokenizer: object = None
    stream: object = None
    model_dir: Optional[Path] = None
    model_id: str = ""
    device: str = "CPU"
    provider: str = "CPUExecutionProvider"
    requested_device: Optional[str] = None
    fell_back: bool = False
    notes: str = ""


def _provider_for_device(device: str) -> str:
    return {
        "NPU": "QNNExecutionProvider",
        "GPU": "DmlExecutionProvider",
        "CPU": "CPUExecutionProvider",
    }.get(device, "CPUExecutionProvider")


def _split_reasoning(raw: str) -> tuple[str, str]:
    """Split Qwen3's <think> block out of the answer."""
    if "<think>" not in raw.lower():
        return "", raw
    match = _THINK_RE.search(raw)
    if match:
        return match.group(0), raw.replace(match.group(0), "").strip()
    open_idx = raw.lower().find("<think>")
    return raw[open_idx:], raw[:open_idx].strip()


class LocalLLM:
    """
    Hardware-routed local text generation.

    Device selection mirrors the vision engine: ask the router where this
    task should run, bind the GenAI model to that device, then report what
    actually happened rather than what we hoped for.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, auto_route: bool = True):
        self._model_id = model_id
        self._auto_route = auto_route
        self._state = _LoadedLLM()
        self._genai = None
        self._route_plan = None
        # Hardware does not change between turns, but detect_hardware() costs
        # ~2s. Probe once, not on every generate() call.
        self._system_prompt_cache: Optional[str] = None

    # ── Model acquisition ────────────────────────────────────────────────────

    @staticmethod
    def model_dir(root: Optional[Path] = None) -> Path:
        from config import settings
        if root is not None:
            return Path(root)
        return settings.MODEL_DIRECTORY / "llm"

    @classmethod
    def is_downloaded(cls, model_id: str = DEFAULT_MODEL, root: Optional[Path] = None) -> bool:
        d = cls.model_dir(root) / model_id
        return (d / "genai_config.json").exists() and (d / "model.onnx").exists()

    @classmethod
    def download(cls, model_id: str = DEFAULT_MODEL, root: Optional[Path] = None,
                 progress: bool = True) -> tuple[bool, str]:
        """
        Fetch a catalog model from Hugging Face.

        Uses only the standard library so the project gains no hard dependency
        on `huggingface_hub` or `requests`.
        """
        meta = CATALOG.get(model_id)
        if meta is None:
            return False, f"Unknown model '{model_id}'. Known: {list(CATALOG)}"

        dest = cls.model_dir(root) / model_id
        if (dest / "genai_config.json").exists() and (dest / "model.onnx").exists():
            return True, f"Model already present at {dest}"

        dest.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{meta['repo']}/resolve/main/{meta['subfolder']}"
        files = ["genai_config.json", "config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "model.onnx"]

        import urllib.request

        for fname in files:
            out = dest / fname
            if out.exists() and out.stat().st_size > 0:
                continue
            if progress:
                log.info(f"Downloading {fname} from {meta['repo']}...")
            try:
                req = urllib.request.Request(
                    f"{base}/{fname}", headers={"User-Agent": "spectra/1.0"})
                with urllib.request.urlopen(req, timeout=180) as resp, open(out, "wb") as fh:
                    total = int(resp.headers.get("Content-Length", 0))
                    got = 0
                    last = 0
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if progress and total and got - last > (total // 20):
                            last = got
                            print(f"\r  {fname}: {got/1048576:.1f} / "
                                  f"{total/1048576:.1f} MB "
                                  f"({got/total*100:.0f}%)", end="", flush=True)
                    if progress:
                        print(f"\r  {fname}: {got/1048576:.1f} MB (done)".ljust(58))
            except Exception as e:
                return False, f"Failed to download {fname}: {e}"

        (dest / "catalog.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        size = sum(f.stat().st_size for f in dest.glob("*") if f.is_file()) / 1048576
        return True, f"Downloaded {meta['name']} to {dest} ({size:.0f} MB)"

    # ── Routing ──────────────────────────────────────────────────────────────

    def _route(self) -> tuple[str, object]:
        """Ask the router where a text-generation task should run."""
        if not self._auto_route:
            return "CPU", None
        try:
            from core.model_manager.manager import Modality, ModelTask
            from core.router.router import (
                LatencyRequirement, WorkloadRequest, get_router,
            )
            from runtime.manager import ExecutionDevice

            req = WorkloadRequest(
                modality=Modality.TEXT,
                task=ModelTask.GENERATION,
                input_type="text",
                latency_requirement=LatencyRequirement.INTERACTIVE,
                offline_required=True,
                model_id_hint=self._model_id,
                preferred_device=ExecutionDevice.NPU,
            )
            plan = get_router().route(req)
            dev = plan.selected_device.value if plan.selected_device else "CPU"
            self._route_plan = plan
            return ("CPU" if dev == "UNKNOWN" else dev), plan
        except Exception as e:
            log.warning(f"LLM routing failed ({e}); defaulting to CPU")
            return "CPU", None

    @property
    def route_plan(self):
        return self._route_plan

    # ── Load / unload ────────────────────────────────────────────────────────

    def load(self, model_dir: Optional[Path] = None) -> tuple[bool, str]:
        """Bind a GenAI model to the routed device."""
        try:
            import onnxruntime_genai as og
        except ImportError:
            return False, ("onnxruntime-genai is not installed. "
                           "Run: pip install onnxruntime-genai")

        self._genai = og
        path = Path(model_dir) if model_dir else self.model_dir() / self._model_id

        if not (path / "genai_config.json").exists():
            return False, f"No model at {path}. Run: python scripts/download_llm.py"

        requested, _plan = self._route()
        self._state = _LoadedLLM(model_id=self._model_id,
                                 requested_device=requested,
                                 model_dir=path)
        self._system_prompt_cache = None

        try:
            model = self._build_model(og, path, requested)
            if model is None:
                return False, "Could not construct model"

            self._state.model = model
            self._state.tokenizer = og.Tokenizer(model)
            try:
                self._state.stream = self._state.tokenizer.create_stream()
            except Exception:
                self._state.stream = None

            # Report what actually bound, not what we requested.
            bound = str(getattr(model, "device_type", requested) or requested).upper()
            self._state.device = bound if bound in ("CPU", "GPU", "NPU") else requested
            self._state.provider = _provider_for_device(self._state.device)

            if self._state.device != requested:
                self._state.fell_back = True
                extra = ("On Snapdragon this usually means onnxruntime-qnn is "
                         "not installed." if requested == "NPU" else "")
                self._state.notes = (
                    f"Requested {requested}, but the model bound to "
                    f"{self._state.device}. {extra}"
                ).strip()

            log.info(f"LLM loaded: {path.name} | requested={requested} | "
                     f"bound={self._state.device}")
            msg = f"LLM ready: {path.name} on {self._state.device}"
            if self._state.fell_back:
                msg += f" [FALLBACK: {self._state.notes}]"
            return True, msg

        except Exception as e:
            log.error(f"Failed to load LLM from {path}: {e}")
            return False, f"LLM load failed: {e}"

    def _build_model(self, og, path: Path, requested: str):
        """
        Construct the model, requesting the routed device.

        onnxruntime-genai selects the EP through `Config.append_provider`. We
        try the requested device first and fall back to plain construction if
        that provider is unavailable in this build.
        """
        if requested and requested != "CPU":
            ep = _provider_for_device(requested)
            try:
                cfg = og.Config(str(path))
                cfg.append_provider(ep)
                model = og.Model(cfg)
                log.info(f"Requested execution provider {ep} for LLM")
                return model
            except Exception as e:
                log.warning(
                    f"Could not bind LLM to {ep} ({e}); using default provider. "
                    f"On Snapdragon, install onnxruntime-qnn for NPU execution."
                )
        return og.Model(str(path))

    def unload(self) -> None:
        self._state = _LoadedLLM()
        self._system_prompt_cache = None

    # ── Prompt construction ──────────────────────────────────────────────────

    def _render_chat(self, msgs: list[dict], thinking: bool) -> Optional[str]:
        """
        Render the chat template with a real Jinja context.

        `apply_chat_template` takes a JSON string and offers no
        `enable_thinking` kwarg in this binding, so we render the model's own
        chat_template.jinja ourselves. Returns None if that is not possible.
        """
        tokenizer = self._state.tokenizer
        payload = json.dumps(msgs)
        tmpl_path = (self._state.model_dir or self.model_dir() / self._model_id) \
            / "chat_template.jinja"

        template_str = None
        if tmpl_path.exists():
            template_str = tmpl_path.read_text(encoding="utf-8")

        # Newest binding: pass the flag through the Jinja context.
        for kwargs in (
            {"template_str": template_str, "add_generation_prompt": True,
             "enable_thinking": thinking},
            {"template_str": template_str, "add_generation_prompt": True},
            {"add_generation_prompt": True},
        ):
            try:
                return tokenizer.apply_chat_template(payload, **kwargs)
            except TypeError:
                continue
            except Exception as e:
                log.debug(f"apply_chat_template variant failed: {e}")
                continue
        return None

    @staticmethod
    def _fallback_chat(prompt: str, thinking: bool) -> str:
        """ChatML fallback used when the template cannot be rendered."""
        text = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        if not thinking:
            text += _EMPTY_THINK_BLOCK
        return text

    def _build_prompt(self, msgs: list[dict], prompt: str, thinking: bool) -> str:
        text = self._render_chat(msgs, thinking)
        if text:
            if not thinking and "<think>" not in text:
                # Template honoured thinking mode: force it off by pre-closing
                # the block the template would have opened.
                text += _EMPTY_THINK_BLOCK
            return text

        # No usable template: build ChatML by hand.
        log.warning("Chat template unavailable; using manual ChatML prompt")
        user_only = [{"role": "user", "content": prompt}]
        parts = []
        for m in msgs:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        text = "".join(parts) + "<|im_start|>assistant\n"
        if not thinking:
            text += _EMPTY_THINK_BLOCK
        del user_only
        return text

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system: Optional[str] = None,
        keep_reasoning: bool = False,
        thinking: bool = False,
    ) -> GenerationResult:
        """Generate a completion for `prompt` on local silicon."""
        if self._state.model is None or self._genai is None:
            ok, msg = self.load()
            if not ok:
                return GenerationResult(success=False, prompt=prompt,
                                        model_id=self._model_id,
                                        error_message=msg)

        og = self._genai
        meta = CATALOG.get(self._model_id, {})
        t0 = time.perf_counter()
        first_token_at: Optional[float] = None
        collected: list[str] = []
        n = 0

        # A small model has no idea where it is running and will happily guess
        # ("running on my hardware"). Give it the truth from SPECTRA's own
        # hardware detection instead of letting it improvise.
        if system is None:
            system = self.system_prompt()

        try:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})

            text = self._build_prompt(msgs, prompt, thinking)
            tokens = self._state.tokenizer.encode(text)

            params = og.GeneratorParams(self._state.model)
            # `max_length` is the TOTAL sequence length in ONNX Runtime GenAI
            # (prompt + generated), not a generation budget. Passing
            # max_tokens straight through would truncate every answer after
            # roughly len(prompt) tokens.
            total_budget = len(tokens) + max_tokens
            _set_search_options(params, max_length=total_budget,
                                temperature=temperature, top_p=top_p,
                                batch_size=1)

            generator = og.Generator(self._state.model, params)
            if hasattr(generator, "append_tokens"):
                generator.append_tokens(tokens)
            else:  # pragma: no cover - older genai
                _legacy_input_ids(params, tokens)

            while not generator.is_done():
                generator.generate_next_token()
                n += 1
                if n == 1:
                    first_token_at = time.perf_counter()
                piece = self._decode_next(generator)
                if piece:
                    collected.append(piece)

            total_ms = (time.perf_counter() - t0) * 1000.0
            ttft_ms = ((first_token_at or t0) - t0) * 1000.0

            raw = "".join(collected).strip()
            reasoning, answer = _split_reasoning(raw)

            notes = self._state.notes
            if reasoning and not answer:
                # Every token went into the think block: surface it rather than
                # returning an empty answer, and say what happened.
                answer = reasoning
                notes = (
                    f"{notes} " if notes else ""
                ) + (f"All {n} tokens went into the model's <think> block and it "
                     f"was cut off before answering. Retry with thinking=True, "
                     f"or raise max_tokens above {total_budget}.")
            elif reasoning:
                notes = (
                    f"{notes} " if notes else ""
                ) + "Answered after a reasoning block; pass keep_reasoning=True to see it."

            result = GenerationResult(
                success=True,
                text=raw if keep_reasoning else answer,
                reasoning=reasoning,
                prompt=prompt,
                model_id=self._model_id,
                model_name=meta.get("name", self._model_id),
                device=self._state.device,
                provider=self._state.provider,
                backend="onnxruntime-genai",
                prompt_tokens=len(tokens),
                completion_tokens=n,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                tokens_per_second=(n / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
                requested_device=self._state.requested_device,
                fell_back_to_cpu=self._state.fell_back,
                notes=notes,
            )
            log.info(f"LLM: {n} tokens in {total_ms:.0f}ms "
                     f"({result.tokens_per_second:.2f} tok/s) on {result.device}")
            return result

        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            return GenerationResult(
                success=False, prompt=prompt, model_id=self._model_id,
                device=self._state.device, provider=self._state.provider,
                error_message=str(e),
            )

    def _decode_next(self, generator) -> str:
        """Decode the newest token, preferring the streaming tokenizer."""
        try:
            new_tokens = generator.get_next_tokens()
            token = new_tokens[0] if len(new_tokens) else None
        except Exception:
            return ""
        if token is None:
            return ""
        if self._state.stream is not None:
            try:
                return self._state.stream.decode(token)
            except Exception:
                pass
        try:
            return self._state.tokenizer.decode([token])
        except Exception:
            return ""

    # ── Introspection ────────────────────────────────────────────────────────

    def refresh_system_prompt(self) -> None:
        """Force re-detection of hardware for the next generation."""
        self._system_prompt_cache = None

    def system_prompt(self) -> str:
        """
        Ground the model in the real machine it is running on.

        Without this a small model answers "it runs on my hardware" or invents
        an NPU. We hand it the detected CPU/GPU/NPU and the device the router
        actually bound, so its self-descriptions are measurable rather than
        decorative.

        Cached: the hardware probe is slow and the answer cannot change
        between turns.
        """
        if self._system_prompt_cache:
            return self._system_prompt_cache

        cpu = "unknown CPU"
        npu = "not present on this machine"
        try:
            from hardware.detector import detect_hardware
            hw = detect_hardware()
            cpu = hw.cpu.name
            npu = hw.npu.name if hw.npu.available else "not present on this machine"
        except Exception as e:
            log.debug(f"hardware detection unavailable for system prompt: {e}")

        prompt = (
            "You are SPECTRA, a fully offline AI assistant running entirely "
            "on the user's own laptop. No cloud services are used.\n"
            "Hardware facts about this machine, as measured by SPECTRA:\n"
            f"- Inference device: {self._state.device} "
            f"(execution provider {self._state.provider})\n"
            f"- Processor: {cpu}\n"
            f"- NPU: {npu}\n"
            "Rules:\n"
            "- Be concise and factual. Answer in at most two sentences unless "
            "asked for more detail.\n"
            "- If asked where or on what hardware you run, reply with the "
            f"exact string '{self._state.device}' and name the execution "
            "provider. Never say 'my hardware' or guess.\n"
            "- Never claim to use a device listed above as absent."
        )
        self._system_prompt_cache = prompt
        return prompt

    @property
    def is_loaded(self) -> bool:
        return self._state.model is not None

    @property
    def device(self) -> str:
        return self._state.device

    @property
    def provider(self) -> str:
        return self._state.provider

    @property
    def requested_device(self) -> Optional[str]:
        return self._state.requested_device

    @property
    def fell_back_to_cpu(self) -> bool:
        return self._state.fell_back


def _set_search_options(params, **options) -> None:
    """Configure generation, tolerating API drift between genai versions."""
    try:
        params.set_search_options(**options)
        return
    except Exception:
        pass
    for k, v in options.items():
        try:
            setattr(params, k, v)
        except Exception:
            pass


def _legacy_input_ids(params, tokens) -> None:
    """Set prompt tokens on pre-0.17 GeneratorParams."""
    try:
        params.input_ids = tokens
    except Exception as e:  # pragma: no cover
        log.warning(f"Could not set input_ids: {e}")
