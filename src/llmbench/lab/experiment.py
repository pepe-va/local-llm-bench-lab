"""Starting points for new experiments in the lab: test presets and machine profiles."""

import re
from dataclasses import dataclass
from pathlib import Path

from llmbench.lab.models import LocalModel


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    help: str
    phase: str
    suites: tuple[str, ...] = ()


PRESETS = (
    Preset(
        "speed", "speed", "llama-bench: prompt processing (pp512) and generation (tg128) "
        "tokens/s, 3 repetitions with a 95% CI.", "bench",
    ),
    Preset(
        "latency", "latency", "llama-server, one request at a time: time to first token, "
        "inter-token latency and tokens/s, with synthetic 512→128 and chat-like prompts.",
        "latency", ("synthetic-512x128.yaml", "synthetic-chat-mix.yaml"),
    ),
    Preset(
        "quality", "Spanish writing", "llama-server: correcting, rewriting and summarising "
        "Spanish text, scored by deterministic checks.", "quality", ("es-writing.yaml",),
    ),
)  # fmt: skip


def default_profile(profiles: list[str]) -> str | None:
    """This machine's profile among `_profiles/*.yaml`: pi5 on a Raspberry Pi, else pc."""
    wanted = "_profiles/pi5.yaml" if _is_raspberry_pi() else "_profiles/pc.yaml"
    usable = [p for p in profiles if not p.endswith("/deterministic.yaml")]
    return wanted if wanted in usable else (usable[0] if usable else None)


def default_ref(profile: Path) -> str:
    """The llama.cpp ref a profile pins (`${env:LLB_LLAMA_REF:-<ref>}` or a literal)."""
    try:
        text = profile.read_text()
    except OSError:
        return "master"
    if m := re.search(r"ref:\s*\$\{env:\w+:-([\w.\-/]+)\}", text):
        return m.group(1)
    if m := re.search(r"^\s*ref:\s*([\w.\-/]+)\s*$", text, re.MULTILINE):
        return m.group(1)
    return "master"


def _is_raspberry_pi() -> bool:
    try:
        return "Raspberry Pi" in Path("/proc/device-tree/model").read_text(errors="replace")
    except OSError:
        return False


def experiment_name(model: LocalModel) -> str:
    name = model.name.removesuffix(".gguf")
    if model.quant.lower() not in name.lower():
        name = f"{name}-{model.quant}"
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")
    return f"lab-{slug}"[:80].rstrip("-.")
