# app.py
"""
ONGEA LABS — ChatGPT-style African Voice Studio (CPU-first)
- ChatGPT-like layout: left sidebar (collapsible), centered chat feed, bottom input bar
- Clean light/dark themes (unique palettes) + smooth transitions
- Voice settings live in a top-right Settings popover (language, voice, speed, pitch)
- Ongea (single) + Batch (line-by-line) without Streamlit radio UI
- CPU-first: no OpenAI/ChatGPT API tokens (runs HF/Meta MMS locally on the server CPU)

Run:
  streamlit run app.py

Train (local):
  python app.py --train --lang swh
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import re
import json
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

import numpy as np


# =========================
# PROJECT SETTINGS
# =========================

HF_DATASET_NAME = "michsethowusu/swahili-words-speech-text-parallel"
HF_DATASET_CONFIG = None
TRAIN_SPLIT = "train"
EVAL_SPLIT = None

DEFAULT_AUDIO_COL = "audio"
DEFAULT_TEXT_COL = "text"

LANGUAGES: Dict[str, str] = {
    "Swahili (Kiswahili)": "swh",
    "English": "eng",
    "Amharic (አማርኛ)": "amh",
    "Arabic (العربية)": "ara",
    "Shona": "sna",
    "Igbo": "ibo",
    "Afrikaans": "afr",

    "Somali (Soomaaliga)": "som",
    "Yoruba (Yorùbá)": "yor",
    "Xhosa (isiXhosa)": "xho",
    "Lingala": "lin",
    "Kongo (Kikongo)": "kon",
    "Luo (Dholuo)": "luo",
    "Gikuyu (Agĩkũyũ)": "kik",
    "Ameru / Meru": "mer",
    "Kamba (Kikamba)": "kam",
    "Ekegusii (Kisii)": "guz",
    "Luhya (Luluhya)": "luy",
    "Kalenjin": "kln",
    "Maasai (Maa)": "mas",
    "Taita / Dawida": "dav",
}

PROJECT_NAME = "ongea-labs-chat-ui"
OUTPUT_DIR = Path("./outputs") / PROJECT_NAME
CONVERTED_DIR = OUTPUT_DIR / "training_checkpoint_with_discriminator"
FINETUNE_REPO = Path("./finetune-hf-vits")

TARGET_SR = 16000
MIN_AUDIO_SEC = 1.0
MAX_AUDIO_SEC = 15.0

LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
MAX_STEPS = 8000
WARMUP_STEPS = 200
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500

LOWERCASE = True
STRIP_PUNCT = False


# =========================
# VOICE LIBRARY
# =========================

VOICE_LIBRARY_BY_LANG: Dict[str, Dict[str, str]] = {
    "swh": {
        "Ongea Swahili — Meta MMS (Neutral)": "facebook/mms-tts-swh",
        "Ongea Swahili — Mozilla Lady (Fine-tuned)": "Benjamin-png/swahili-mms-tts-mozilla-lady-voice-finetuned",
        "Ongea Swahili — Studio (Fine-tuned)": "Benjamin-png/swahili-mms-tts-finetuned",
        "Ongea Swahili — OpenBible Narrator": "bookbot/vits-base-sw-KE-OpenBible",
        "Ongea Swahili — SALAMA (Prosody-rich)": "EYEDOL/SALAMA_TTS",
    },
    "eng": {"Ongea English — Meta MMS": "facebook/mms-tts-eng"},
    "amh": {"Ongea Amharic — Meta MMS": "facebook/mms-tts-amh"},
    "ara": {"Ongea Arabic — Meta MMS": "facebook/mms-tts-ara"},
    "sna": {"Ongea Shona — Meta MMS": "facebook/mms-tts-sna"},
    "ibo": {"Ongea Igbo — Meta MMS": "facebook/mms-tts-ibo"},
    "afr": {"Ongea Afrikaans — Meta MMS": "facebook/mms-tts-afr"},

    "som": {"Ongea Somali — Meta MMS": "facebook/mms-tts-som"},
    "yor": {"Ongea Yoruba — Meta MMS": "facebook/mms-tts-yor"},
    "xho": {"Ongea Xhosa — Meta MMS": "facebook/mms-tts-xho"},
    "lin": {"Ongea Lingala — Meta MMS": "facebook/mms-tts-lin"},
    "kon": {"Ongea Kongo — Meta MMS": "facebook/mms-tts-kon"},
    "luo": {
        "Ongea Luo — Meta MMS": "facebook/mms-tts-luo",
        "Ongea Luo — CLEAR YourTTS (Coqui)": "coqui:CLEAR-Global/YourTTS-Luo",
        "Ongea Luo — CLEAR XTTS (Coqui)": "coqui:CLEAR-Global/XTTS-Luo",
    },
    "kik": {"Ongea Gikuyu — Meta MMS": "facebook/mms-tts-kik"},
    "mer": {"Ongea Meru — Meta MMS": "facebook/mms-tts-mer"},
    "kam": {"Ongea Kamba — Meta MMS": "facebook/mms-tts-kam"},
    "guz": {"Ongea Ekegusii — Meta MMS": "facebook/mms-tts-guz"},
    "luy": {"Ongea Luhya — Meta MMS": "facebook/mms-tts-luy"},
    "kln": {"Ongea Kalenjin — Meta MMS": "facebook/mms-tts-kln"},
    "mas": {"Ongea Maasai — Meta MMS": "facebook/mms-tts-mas"},
    "dav": {"Ongea Taita/Dawida — Meta MMS": "facebook/mms-tts-dav"},
}


# =========================
# UTILITIES
# =========================

def run(cmd, cwd=None, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)

def ensure_repo():
    if FINETUNE_REPO.exists():
        return
    run(["git", "clone", "https://github.com/ylacombe/finetune-hf-vits.git", str(FINETUNE_REPO)])

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = t.strip()
    if LOWERCASE:
        t = t.lower()
    t = re.sub(r"\s+", " ", t)
    if STRIP_PUNCT:
        t = re.sub(r"[^\w\s']", "", t)
    return t


# =========================
# TRAINING (lazy imports)
# =========================

def detect_columns(ds) -> Tuple[str, str]:
    cols = set(ds.column_names)
    audio_col = DEFAULT_AUDIO_COL if DEFAULT_AUDIO_COL in cols else None
    text_col = DEFAULT_TEXT_COL if DEFAULT_TEXT_COL in cols else None
    if audio_col is None:
        for c in cols:
            if "audio" in c or "speech" in c or "wav" in c:
                audio_col = c
                break
    if text_col is None:
        for c in cols:
            if "text" in c or "transcript" in c or "sentence" in c:
                text_col = c
                break
    if audio_col is None or text_col is None:
        raise ValueError(f"Could not auto-detect columns. Found: {cols}")
    return audio_col, text_col

def load_and_prepare_dataset() -> Tuple[Any, Optional[Any], str, str]:
    from datasets import load_dataset, Audio
    ds_train = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=TRAIN_SPLIT)
    ds_eval = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=EVAL_SPLIT) if EVAL_SPLIT else None

    audio_col, text_col = detect_columns(ds_train)
    ds_train = ds_train.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))
    if ds_eval is not None:
        ds_eval = ds_eval.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    def _norm(ex):
        ex[text_col] = clean_text(ex[text_col])
        return ex

    ds_train = ds_train.map(_norm)
    if ds_eval is not None:
        ds_eval = ds_eval.map(_norm)

    def _keep(ex):
        a = ex[audio_col]
        if a is None or a.get("array") is None:
            return False
        dur = len(a["array"]) / a["sampling_rate"]
        if dur < MIN_AUDIO_SEC or dur > MAX_AUDIO_SEC:
            return False
        if ex[text_col] is None or ex[text_col].strip() == "":
            return False
        return True

    ds_train = ds_train.filter(_keep)
    if ds_eval is not None:
        ds_eval = ds_eval.filter(_keep)

    return ds_train, ds_eval, audio_col, text_col

def maybe_convert_discriminator(lang_code: str) -> str:
    lang_dir = CONVERTED_DIR / lang_code
    if (lang_dir / "config.json").exists():
        return str(lang_dir)
    ensure_repo()
    lang_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python", "convert_original_discriminator_checkpoint.py",
        "--language_code", lang_code,
        "--pytorch_dump_folder_path", str(lang_dir),
    ], cwd=FINETUNE_REPO)
    return str(lang_dir)

def build_finetune_config(model_path: str, audio_col: str, text_col: str, lang_code: str) -> Dict[str, Any]:
    import torch
    return {
        "project_name": f"{PROJECT_NAME}-{lang_code}",
        "model_name_or_path": model_path,
        "output_dir": str(OUTPUT_DIR / lang_code),
        "push_to_hub": False,
        "dataset_name": HF_DATASET_NAME,
        "dataset_config_name": HF_DATASET_CONFIG,
        "train_split_name": TRAIN_SPLIT,
        "eval_split_name": EVAL_SPLIT,
        "audio_column_name": audio_col,
        "text_column_name": text_col,
        "sampling_rate": TARGET_SR,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "logging_steps": LOGGING_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "do_train": True,
        "do_eval": bool(EVAL_SPLIT),
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "max_audio_length_in_seconds": MAX_AUDIO_SEC,
        "min_audio_length_in_seconds": MIN_AUDIO_SEC,
    }

def launch_training(lang_code: str):
    _, _, audio_col, text_col = load_and_prepare_dataset()
    model_path = maybe_convert_discriminator(lang_code)

    out_dir = OUTPUT_DIR / lang_code
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_finetune_config(model_path, audio_col, text_col, lang_code)
    cfg_path = out_dir / "finetune_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    ensure_repo()
    train_script = FINETUNE_REPO / "run_vits_finetuning.py"
    run(["accelerate", "launch", str(train_script), "--config", str(cfg_path)], cwd=FINETUNE_REPO)


# =========================
# TTS LOADING + SYNTHESIS
# =========================

BASE_MMS_REPO = "facebook/mms-tts"
COQUI_PREFIX = "coqui:"

@dataclass
class VoiceBundle:
    engine: str
    processor: Any = None
    model: Any = None
    sr: int = TARGET_SR
    model_id: str = ""
    lang_code: str = ""

def _get_model_classes():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForTextToWaveform
        return AutoProcessor, AutoModelForTextToWaveform
    except Exception:
        from transformers import VitsModel
        return AutoProcessor, VitsModel

def _encode_inputs(processor, text: str):
    import torch
    try:
        inputs = processor(text=text, return_tensors="pt")
    except TypeError:
        inputs = processor(text=text, return_tensors="pt", normalize=False)
    ids = inputs.get("input_ids", None)
    if ids is not None and isinstance(ids, torch.Tensor) and (ids.numel() == 0 or ids.shape[-1] == 0):
        raise ValueError("Tokenizer produced empty input_ids.")
    return inputs

def _maybe_unidecode(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text

def _safe_load_hf_vits(model_id: str, lang_code: Optional[str] = None) -> VoiceBundle:
    import torch
    AutoProcessor, ModelClass = _get_model_classes()
    last_err = None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = ModelClass.from_pretrained(model_id, low_cpu_mem_usage=False, device_map=None)
        if any(getattr(p, "is_meta", False) for p in model.parameters()):
            raise RuntimeError("Model loaded with meta tensors.")
        model.to("cpu")
        model.eval()
        return VoiceBundle(
            engine="hf_vits",
            processor=processor,
            model=model,
            sr=int(getattr(processor, "sampling_rate", TARGET_SR)),
            model_id=model_id,
            lang_code=lang_code or "",
        )
    except Exception as e:
        last_err = e

    if lang_code:
        try:
            sub = f"models/{lang_code}"
            processor = AutoProcessor.from_pretrained(BASE_MMS_REPO, subfolder=sub)
            model = ModelClass.from_pretrained(
                BASE_MMS_REPO, subfolder=sub, low_cpu_mem_usage=False, device_map=None
            )
            if any(getattr(p, "is_meta", False) for p in model.parameters()):
                raise RuntimeError("Model loaded with meta tensors.")
            model.to("cpu")
            model.eval()
            return VoiceBundle(
                engine="hf_vits",
                processor=processor,
                model=model,
                sr=int(getattr(processor, "sampling_rate", TARGET_SR)),
                model_id=model_id,
                lang_code=lang_code,
            )
        except Exception as e:
            last_err = e

    raise last_err

def _safe_load_coqui_hf(model_id: str) -> VoiceBundle:
    real_id = model_id[len(COQUI_PREFIX):].strip()
    try:
        from huggingface_hub import snapshot_download
        from TTS.api import TTS as CoquiTTS
    except Exception as e:
        raise RuntimeError("Coqui requested but not installed. pip install TTS huggingface_hub") from e

    local_dir = Path(snapshot_download(repo_id=real_id))
    cfgs = list(local_dir.rglob("config.json")) + list(local_dir.rglob("*config*.json"))
    if not cfgs:
        raise RuntimeError(f"No config.json found in {real_id}")
    config_path = str(cfgs[0])

    ckpts = []
    for ext in ("*.pth", "*.pt", "*.bin", "*.safetensors"):
        ckpts += list(local_dir.rglob(ext))
    ckpts = [p for p in ckpts if p.is_file()]
    if not ckpts:
        raise RuntimeError(f"No checkpoint found in {real_id}")

    model_path = str(ckpts[0])
    tts = CoquiTTS(model_path=model_path, config_path=config_path, progress_bar=False, gpu=False)

    out_sr = TARGET_SR
    try:
        out_sr = int(getattr(tts.synthesizer, "output_sample_rate", TARGET_SR))
    except Exception:
        pass

    return VoiceBundle(engine="coqui", model=tts, sr=out_sr, model_id=model_id)

def _safe_load_model(model_id: str, lang_code: Optional[str] = None) -> VoiceBundle:
    if model_id.startswith(COQUI_PREFIX):
        return _safe_load_coqui_hf(model_id)
    return _safe_load_hf_vits(model_id, lang_code=lang_code)

def _to_1d_float32(audio) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().float().numpy()
    except Exception:
        pass
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    return audio

def synthesize_raw(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    import numpy as np

    text = clean_text(text)
    if not text:
        raise ValueError("Empty text.")

    if bundle.engine == "coqui":
        try:
            wave = bundle.model.tts(text)
        except Exception:
            wave = bundle.model.tts(_maybe_unidecode(text))
        audio = _to_1d_float32(wave)
        sr = int(bundle.sr or TARGET_SR)
        if audio.size == 0:
            raise ValueError("Coqui returned empty audio.")
        m = float(np.max(np.abs(audio)))
        if m > 1.0:
            audio = audio / m
        return np.clip(audio, -1.0, 1.0), sr

    processor, model = bundle.processor, bundle.model
    inputs = _encode_inputs(processor, text)
    import torch
    with torch.no_grad():
        out = model(**inputs)

    wave = out["waveform"] if isinstance(out, dict) and "waveform" in out else getattr(out, "waveform", None) or out[0]
    audio = _to_1d_float32(wave)
    sr = int(getattr(processor, "sampling_rate", TARGET_SR))

    if audio.size == 0:
        raise ValueError("Model returned empty audio.")
    m = float(np.max(np.abs(audio)))
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), sr

def split_by_punctuation(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"\.\.\.+", "…", text)
    parts = re.findall(r"([^,.;:!?…]+)([,.;:!?…]?)", text)
    out = []
    for chunk, punct in parts:
        c = chunk.strip()
        if c:
            out.append((c, punct))
    return out

def synthesize_human(bundle: VoiceBundle, text: str) -> Tuple[np.ndarray, int]:
    chunks = split_by_punctuation(text)
    if not chunks:
        raise ValueError("Empty text.")
    audios = []
    sr_final = bundle.sr or TARGET_SR
    pause = {",": 0.18, ";": 0.22, ":": 0.22, ".": 0.38, "!": 0.42, "?": 0.42, "…": 0.55}

    for chunk_text, punct in chunks:
        a, sr = synthesize_raw(bundle, chunk_text)
        sr_final = sr_final or sr
        audios.append(a)
        dur = pause.get(punct, 0.0)
        if dur > 0:
            audios.append(np.zeros(int(sr * dur), dtype=np.float32))

    audio = np.concatenate(audios) if len(audios) > 1 else audios[0]
    m = float(np.max(np.abs(audio))) if audio.size else 1.0
    if m > 1.0:
        audio = audio / m
    return np.clip(audio, -1.0, 1.0), int(sr_final)

def apply_tone(audio: np.ndarray, sr: int, speed: float, pitch_semitones: float) -> np.ndarray:
    try:
        import librosa
        y = audio.astype(np.float32)
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)
        if pitch_semitones != 0.0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)
        m = float(np.max(np.abs(y))) if y.size else 1.0
        if m > 1.0:
            y = y / m
        return np.clip(y, -1.0, 1.0)
    except Exception:
        return audio

def write_wav(path: Path, audio: np.ndarray, sr: int):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16", format="WAV")


# =========================
# CHATGPT-LIKE STREAMLIT UI
# =========================

DISCLAIMER_TEXT = "Ongea can make mistakes. Always listen and double check before you export."

def _init_state(st):
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("sidebar_open", True)
    st.session_state.setdefault("mode", "Ongea")
    st.session_state.setdefault("lang_name", "Swahili (Kiswahili)")
    st.session_state.setdefault("voice_name", None)
    st.session_state.setdefault("speed", 1.0)
    st.session_state.setdefault("pitch", 0.0)

    st.session_state.setdefault("chats", [])
    st.session_state.setdefault("active_chat_id", None)

    st.session_state.setdefault("draft_ongea", "")
    st.session_state.setdefault("draft_batch", "")

    if not st.session_state.chats:
        _new_chat(st)

def _new_chat(st):
    cid = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    chat = {
        "id": cid,
        "title": "New voice session",
        "created": datetime.now().isoformat(),
        "messages": [],
    }
    st.session_state.chats.append(chat)
    st.session_state.active_chat_id = cid
    st.session_state.draft_ongea = ""
    st.session_state.draft_batch = ""

def _get_active_chat(st):
    cid = st.session_state.active_chat_id
    for c in st.session_state.chats:
        if c["id"] == cid:
            return c
    _new_chat(st)
    return st.session_state.chats[-1]

def get_voices_for(lang_code: str, lang_name: str):
    voices = VOICE_LIBRARY_BY_LANG.get(lang_code, {})
    if not voices:
        voices = {f"Ongea {lang_name} — Meta MMS": f"facebook/mms-tts-{lang_code}"}
    return voices

def get_voice_loader():
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def load_voice(model_id: str, lang_code: str):
        return _safe_load_model(model_id, lang_code=lang_code)

    return load_voice


def inject_css(st, theme: str, sidebar_open: bool):
    sbw = 292
    sb_tx = "0%" if sidebar_open else "-110%"

    st.markdown(f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stHeader"], [data-testid="stDecoration"] {{
  display:none !important;
}}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarToggleButton"],
button[title="Open sidebar"],
button[title="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"] {{
  display:none !important;
}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --font: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;

  --l-bg: #f7f7f5;
  --l-surface: #ffffff;
  --l-surface2: #fbfbf8;
  --l-text: #111214;
  --l-muted: rgba(17,18,20,0.62);
  --l-border: rgba(17,18,20,0.10);

  --d-bg: #0b0d12;
  --d-surface: rgba(18, 22, 32, 0.78);
  --d-surface2: rgba(18, 22, 32, 0.62);
  --d-text: #eaf0ff;
  --d-muted: rgba(234,240,255,0.68);
  --d-border: rgba(234,240,255,0.12);

  --accentA: #6b66ff;
  --accentB: #19b6ad;
  --accentC: #ff4d7d;

  --radius: 18px;
  --shadow: 0 18px 40px rgba(0,0,0,{"0.10" if theme=="light" else "0.40"});
  --shadow2: 0 10px 24px rgba(0,0,0,{"0.08" if theme=="light" else "0.30"});
}}

:root {{
  --bg: {"var(--l-bg)" if theme=="light" else "var(--d-bg)"};
  --surface: {"var(--l-surface)" if theme=="light" else "var(--d-surface)"};
  --surface2: {"var(--l-surface2)" if theme=="light" else "var(--d-surface2)"};
  --text: {"var(--l-text)" if theme=="light" else "var(--d-text)"};
  --muted: {"var(--l-muted)" if theme=="light" else "var(--d-muted)"};
  --border: {"var(--l-border)" if theme=="light" else "var(--d-border)"};
}}

html, body {{
  font-family: var(--font) !important;
  background: transparent !important;
}}

[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(900px 520px at 10% 0%, rgba(107,102,255,{"0.18" if theme=="light" else "0.14"}) 0%, transparent 60%),
    radial-gradient(820px 520px at 90% 8%, rgba(25,182,173,{"0.16" if theme=="light" else "0.12"}) 0%, transparent 60%),
    radial-gradient(760px 520px at 70% 90%, rgba(255,77,125,{"0.14" if theme=="light" else "0.10"}) 0%, transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg)) !important;
}}

.block-container {{
  max-width: 860px !important;
  padding-top: 1.05rem !important;
  padding-bottom: 7.9rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
  transform: translateX(0px) !important;
}}

[data-testid="stHorizontalBlock"],
[data-testid="stColumn"] {{
  overflow: visible !important;
}}

[data-testid="stSidebar"] {{
  position: fixed !important;
  left: 0; top: 0;
  height: 100vh !important;
  width: {sbw}px !important;
  transform: translateX({sb_tx}) !important;
  transition: transform 260ms ease;
  z-index: 999 !important;
  border-right: 1px solid var(--border) !important;
  background: var(--surface2) !important;
  box-shadow: var(--shadow2);
}}
[data-testid="stSidebar"] > div {{
  background: transparent !important;
  padding-top: 0.6rem !important;
}}

.oge-topbar {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 0.75rem;
  padding: 0.15rem 0.00rem 0.15rem 0.00rem;
}}

.oge-brand {{
  display:flex; align-items:center; gap: 0.7rem;
}}
.oge-logo {{
  width: 40px; height: 40px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(25,182,173,0.30), rgba(107,102,255,0.22));
  border: 1px solid var(--border);
  display:flex; align-items:center; justify-content:center;
}}
.oge-brandtext {{
  font-weight: 850;
  letter-spacing: -0.02em;
  line-height: 1.05;
}}
.oge-subtext {{
  font-size: 0.88rem;
  color: var(--muted) !important;
  margin-top: 0.1rem;
}}

.oge-actions {{
  display:flex;
  align-items:center;
  justify-content:flex-end;
  gap: 0.30rem;
}}

.oge-iconbtn .stButton>button {{
  width: 42px !important;
  height: 42px !important;
  border-radius: 12px !important;

  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;

  padding: 0 !important;
  font-size: 1.18rem !important;
  color: var(--text) !important;

  transition: transform 150ms ease, background 150ms ease, filter 150ms ease;
}}
.oge-iconbtn .stButton>button:hover {{
  transform: translateY(-1px);
  background: rgba(107,102,255,{"0.10" if theme=="light" else "0.14"}) !important;
  filter: brightness(1.02);
}}
.oge-iconbtn .stButton>button:active {{
  transform: translateY(0px);
}}

.oge-pop {{
  display:flex;
  justify-content:flex-end;
}}
.oge-pop .stButton>button {{
  width: 42px !important;
  height: 42px !important;
  border-radius: 12px !important;

  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;

  padding: 0 !important;
  font-size: 1.18rem !important;
  color: var(--text) !important;
  white-space: nowrap !important;

  transition: transform 150ms ease, background 150ms ease, filter 150ms ease;
}}
.oge-pop .stButton>button:hover {{
  transform: translateY(-1px);
  background: rgba(25,182,173,{"0.10" if theme=="light" else "0.14"}) !important;
}}

.oge-hero {{
  margin-top: 5.4rem;
  text-align: center;
  animation: oge-fadeup 520ms ease;
}}
.oge-hero h1 {{
  font-size: 2.2rem;
  font-weight: 860;
  letter-spacing: -0.03em;
  margin: 0 0 0.6rem 0;
}}
.oge-hero p {{
  margin: 0 auto;
  max-width: 680px;
  color: var(--muted) !important;
  line-height: 1.55;
  font-size: 1.02rem;
}}
.oge-langloop {{
  margin-top: 0.85rem;
  display:flex;
  justify-content:center;
  gap: 0.45rem;
  flex-wrap: wrap;
}}
.oge-chip {{
  padding: 0.32rem 0.62rem;
  border-radius: 999px;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--muted) !important;
  font-weight: 700;
  font-size: 0.86rem;
  animation: oge-chip 2.8s infinite ease-in-out;
}}
.oge-chip:nth-child(2){{ animation-delay: 0.15s; }}
.oge-chip:nth-child(3){{ animation-delay: 0.30s; }}
.oge-chip:nth-child(4){{ animation-delay: 0.45s; }}
.oge-chip:nth-child(5){{ animation-delay: 0.60s; }}
.oge-chip:nth-child(6){{ animation-delay: 0.75s; }}
.oge-chip:nth-child(7){{ animation-delay: 0.90s; }}

@keyframes oge-chip {{
  0%,100% {{ transform: translateY(0); opacity: 0.78; }}
  50% {{ transform: translateY(-3px); opacity: 1; }}
}}
@keyframes oge-fadeup {{
  from {{ opacity:0; transform: translateY(10px); }}
  to {{ opacity:1; transform: translateY(0); }}
}}
.oge-grad {{
  background: linear-gradient(90deg, var(--accentB), var(--accentA), var(--accentC));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
}}

.oge-feed {{
  margin-top: 0.55rem;
  display:flex;
  flex-direction: column;
  gap: 0.75rem;
}}
.oge-msg {{
  display:flex;
  width: 100%;
  animation: oge-in 240ms ease;
}}
.oge-msg.user {{ justify-content: flex-end; }}
.oge-msg.assistant {{ justify-content: flex-start; }}

.oge-bubble {{
  max-width: 100%;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: var(--surface);
  box-shadow: var(--shadow2);
  padding: 0.85rem 0.95rem;
}}
.oge-msg.user .oge-bubble {{
  background: linear-gradient(135deg, rgba(107,102,255,{"0.12" if theme=="light" else "0.22"}), rgba(25,182,173,{"0.10" if theme=="light" else "0.16"}));
}}

@keyframes oge-in {{
  from {{ opacity:0; transform: translateY(6px); }}
  to {{ opacity:1; transform: translateY(0); }}
}}

audio {{
  width: 100% !important;
  border-radius: 999px !important;
}}

.oge-inputwrap {{
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;

  padding: 0.85rem 0.9rem 0.9rem 0.9rem;

  display:flex;
  flex-direction: column;
  align-items:center;
  gap: 0.45rem;

  pointer-events:none;
  z-index: 998;
}}

.oge-inputcard {{
  width: min(860px, calc(100% - 1.2rem));
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 0.6rem 0.6rem;
  pointer-events:auto;
  backdrop-filter: blur(10px);
}}

.oge-disclaimer {{
  width: min(860px, calc(100% - 1.2rem));
  text-align:center;
  font-size: 0.82rem;
  color: var(--muted);
  font-weight: 650;
  pointer-events:none;
  opacity: {"0.78" if theme=="light" else "0.74"};
}}

.oge-send .stButton>button {{
  width: 48px !important;
  height: 44px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(25,182,173,0.35) !important;
  background: linear-gradient(135deg, rgba(25,182,173,0.22), rgba(107,102,255,0.18)) !important;
  font-weight: 900 !important;
}}
.oge-send .stButton>button:hover {{ transform: translateY(-1px); }}

.oge-modechip {{
  display:flex;
  justify-content:center;
  gap: 0.4rem;
  margin-bottom: 0.35rem;
  width: 100%;
}}
.oge-modechip span {{
  font-size: 0.82rem;
  font-weight: 900;
  color: var(--muted) !important;
  padding: 0.22rem 0.62rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface2);
}}
.oge-modechip span.active {{
  border-color: rgba(25,182,173,0.40);
  background: linear-gradient(135deg, rgba(25,182,173,0.18), rgba(107,102,255,0.14));
}}

textarea {{
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  font-size: 1.02rem !important;
  line-height: 1.35 !important;
}}
textarea::placeholder {{ color: rgba(120,130,150,0.72) !important; }}

[data-testid="stMarkdownContainer"] p {{ margin: 0.25rem 0 0.35rem 0; }}
</style>
""", unsafe_allow_html=True)


def sidebar_view(st):
    sb = st.sidebar

    sb.markdown("""
<div style="padding:0.35rem 0.65rem 0.25rem 0.65rem;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg, rgba(25,182,173,0.30), rgba(107,102,255,0.22));
    border:1px solid var(--border);display:flex;align-items:center;justify-content:center;">🎙️</div>
    <div>
      <div style="font-weight:900;line-height:1.0;">Ongea Labs</div>
      <div style="font-size:0.85rem;color:var(--muted);font-weight:650;margin-top:2px;">Voice Sessions</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if sb.button("＋  New session", use_container_width=True):
        _new_chat(st)
        st.rerun()

    sb.text_input("Search sessions", value="", key="sb_search")

    sb.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    sb.markdown("<div style='padding:0 0.65rem;color:var(--muted);font-weight:800;'>Sessions</div>", unsafe_allow_html=True)
    sb.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    active = st.session_state.active_chat_id
    for chat in reversed(st.session_state.chats[-30:]):
        title = chat["title"]
        is_active = (chat["id"] == active)
        label = ("● " if is_active else "○ ") + title
        if sb.button(label, use_container_width=True, key=f"chat_{chat['id']}"):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    sb.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    sb.markdown("<div style='padding:0.6rem 0.65rem;border-top:1px solid var(--border);color:var(--muted);font-size:0.85rem;'>CPU-first • No API tokens</div>", unsafe_allow_html=True)


def topbar(st):
    left, right = st.columns([0.72, 0.28], gap="small")

    with left:
        st.markdown("""
<div class="oge-topbar">
  <div class="oge-brand">
    <div class="oge-logo">🎙️</div>
    <div>
      <div class="oge-brandtext">Ongea Labs</div>
      <div class="oge-subtext">Generate speech in African & global languages</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with right:
        # Icons only, no boxy cards, and popover won't get clipped
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")

        with c1:
            st.markdown('<div class="oge-iconbtn">', unsafe_allow_html=True)
            if st.button("☰", key="btn_sb", help="Toggle sidebar"):
                st.session_state.sidebar_open = not st.session_state.sidebar_open
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="oge-iconbtn">', unsafe_allow_html=True)
            if st.button("🌙" if st.session_state.theme == "light" else "☀️", key="btn_theme", help="Toggle theme"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="oge-pop">', unsafe_allow_html=True)
            pop = st.popover("⚙️", help="Settings")
            st.markdown("</div>", unsafe_allow_html=True)
            with pop:
                settings_panel(st)


def settings_panel(st):
    st.markdown("<div style='font-weight:900;margin-bottom:0.35rem;'>Mode</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        if st.button("🎙️ Ongea", use_container_width=True):
            st.session_state.mode = "Ongea"
            st.rerun()
    with m2:
        if st.button("🧾 Batch", use_container_width=True):
            st.session_state.mode = "Batch"
            st.rerun()

    st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

    lang_keys = list(LANGUAGES.keys())
    st.session_state.lang_name = st.selectbox(
        "Language",
        lang_keys,
        index=lang_keys.index(st.session_state.lang_name) if st.session_state.lang_name in lang_keys else 0,
        key="lang_select",
    )
    lang_code = LANGUAGES[st.session_state.lang_name]

    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_names = list(voices.keys())
    if st.session_state.voice_name not in voice_names:
        st.session_state.voice_name = voice_names[0]
    st.session_state.voice_name = st.selectbox(
        "Voice / Model",
        voice_names,
        index=voice_names.index(st.session_state.voice_name),
        key="voice_select",
    )

    st.session_state.speed = st.slider("Speed", 0.75, 1.50, float(st.session_state.speed), 0.05)
    st.session_state.pitch = st.slider("Pitch (semitones)", -4.0, 4.0, float(st.session_state.pitch), 0.5)
    st.caption("Tip: punctuation improves prosody.")


def hero_empty_state(st):
    st.markdown("""
<div class="oge-hero">
  <h1><span class="oge-grad">Let’s generate speech</span> — clean, fast, and natural.</h1>
  <p>
    Type your script, pick a voice, and export WAV clips.
    Runs on CPU compute (no ChatGPT/OpenAI token usage).
  </p>
  <div class="oge-langloop">
    <span class="oge-chip">Kiswahili</span>
    <span class="oge-chip">English</span>
    <span class="oge-chip">አማርኛ</span>
    <span class="oge-chip">العربية</span>
    <span class="oge-chip">Shona</span>
    <span class="oge-chip">Igbo</span>
    <span class="oge-chip">Afrikaans</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_feed(st, chat):
    msgs = chat["messages"]
    if not msgs:
        hero_empty_state(st)
        return

    st.markdown('<div class="oge-feed">', unsafe_allow_html=True)

    for msg in msgs:
        role = msg.get("role", "assistant")
        text = msg.get("text", "")

        if role == "user":
            st.markdown(f"""
<div class="oge-msg user">
  <div class="oge-bubble">
    {escape_html(text).replace("\\n", "<br/>")}
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="oge-msg assistant">
  <div class="oge-bubble" style="width:100%;">
    <div style="font-weight:900;margin-bottom:0.35rem;">Ongea Output</div>
    <div style="color:var(--muted);font-weight:650;margin-bottom:0.55rem;">
      {escape_html(msg.get("meta","")).replace("\\n","<br/>")}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            if msg.get("audio_path"):
                st.audio(msg["audio_path"], format="audio/wav")
                st.download_button(
                    "Download WAV",
                    data=Path(msg["audio_path"]).read_bytes(),
                    file_name=Path(msg["audio_path"]).name,
                    mime="audio/wav",
                    use_container_width=True,
                    key=f"dl_{msg.get('id','x')}",
                )

    st.markdown("</div>", unsafe_allow_html=True)


def input_bar(st):
    chat = _get_active_chat(st)

    mode = st.session_state.mode
    active_ongea = "active" if mode == "Ongea" else ""
    active_batch = "active" if mode == "Batch" else ""

    st.markdown(f"""
<div class="oge-inputwrap">
  <div class="oge-inputcard">
    <div class="oge-modechip">
      <span class="{active_ongea}">Ongea</span>
      <span class="{active_batch}">Batch</span>
    </div>
""", unsafe_allow_html=True)

    c_in, c_send = st.columns([0.86, 0.14], gap="small")

    if mode == "Ongea":
        with c_in:
            st.session_state.draft_ongea = st.text_area(
                "Ask",
                value=st.session_state.draft_ongea,
                height=95,
                placeholder="Type your text… (Shift+Enter for new line)",
                label_visibility="collapsed",
                key="input_ongea",
            )
        with c_send:
            st.markdown('<div class="oge-send">', unsafe_allow_html=True)
            send = st.button("➤", use_container_width=True, key="send_ongea", help="Generate speech")
            st.markdown("</div>", unsafe_allow_html=True)

        if send:
            _handle_ongea_send(st, chat)

    else:
        with c_in:
            st.session_state.draft_batch = st.text_area(
                "Batch",
                value=st.session_state.draft_batch,
                height=110,
                placeholder="Paste multiple lines… (1 line = 1 clip)",
                label_visibility="collapsed",
                key="input_batch",
            )
        with c_send:
            st.markdown('<div class="oge-send">', unsafe_allow_html=True)
            send = st.button("➤", use_container_width=True, key="send_batch", help="Generate batch clips")
            st.markdown("</div>", unsafe_allow_html=True)

        if send:
            _handle_batch_send(st, chat)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ChatGPT-style tiny disclaimer (always visible, no scroll, no box)
    st.markdown(f'<div class="oge-disclaimer">{escape_html(DISCLAIMER_TEXT)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _handle_ongea_send(st, chat):
    text = (st.session_state.draft_ongea or "").strip()
    if not text:
        st.warning("Type some text first.")
        return

    if chat["title"] == "New voice session":
        chat["title"] = (text[:26] + "…") if len(text) > 26 else text

    chat["messages"].append({"role": "user", "text": text})

    lang_code = LANGUAGES.get(st.session_state.lang_name, "swh")
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_name = st.session_state.voice_name or list(voices.keys())[0]
    model_id = voices[voice_name]

    load_voice = get_voice_loader()

    try:
        with st.spinner("Loading voice model…"):
            bundle = load_voice(model_id, lang_code)

        with st.spinner("Generating speech…"):
            audio, sr = synthesize_human(bundle, text)
            audio = apply_tone(audio, sr, speed=st.session_state.speed, pitch_semitones=st.session_state.pitch)

        out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_speech_{len(chat['messages']):04d}.wav"
        write_wav(out_wav, audio, sr)

        meta = f"{st.session_state.lang_name} • {voice_name} • {datetime.now().strftime('%H:%M:%S')}"
        chat["messages"].append({
            "role": "assistant",
            "id": f"as_{len(chat['messages'])}",
            "text": "",
            "audio_path": str(out_wav),
            "meta": meta,
        })

        st.session_state.draft_ongea = ""
        st.rerun()

    except Exception as e:
        chat["messages"].append({"role": "assistant", "text": f"⚠️ Generation failed: {e}", "meta": "Error"})
        st.exception(e)
        st.rerun()


def _handle_batch_send(st, chat):
    raw = (st.session_state.draft_batch or "").strip()
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if not lines:
        st.warning("Paste one sentence per line.")
        return

    if chat["title"] == "New voice session":
        first = lines[0]
        chat["title"] = (first[:26] + "…") if len(first) > 26 else first

    chat["messages"].append({"role": "user", "text": "Batch:\n" + "\n".join(lines)})

    lang_code = LANGUAGES.get(st.session_state.lang_name, "swh")
    voices = get_voices_for(lang_code, st.session_state.lang_name)
    voice_name = st.session_state.voice_name or list(voices.keys())[0]
    model_id = voices[voice_name]
    load_voice = get_voice_loader()

    try:
        with st.spinner("Loading voice model…"):
            bundle = load_voice(model_id, lang_code)

        with st.spinner("Generating batch clips…"):
            for i, ln in enumerate(lines, start=1):
                audio, sr = synthesize_human(bundle, ln)
                audio = apply_tone(audio, sr, speed=st.session_state.speed, pitch_semitones=st.session_state.pitch)

                out_wav = OUTPUT_DIR / "app_outputs" / f"{lang_code}_batch_{len(chat['messages']):04d}_{i:02d}.wav"
                write_wav(out_wav, audio, sr)

                meta = f"Line {i}/{len(lines)} • {st.session_state.lang_name} • {voice_name}"
                chat["messages"].append({
                    "role": "assistant",
                    "id": f"as_{len(chat['messages'])}",
                    "text": "",
                    "audio_path": str(out_wav),
                    "meta": meta,
                })

        st.session_state.draft_batch = ""
        st.rerun()

    except Exception as e:
        chat["messages"].append({"role": "assistant", "text": f"⚠️ Batch failed: {e}", "meta": "Error"})
        st.exception(e)
        st.rerun()


def escape_html(s: str) -> str:
    s = s or ""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def run_app():
    import streamlit as st
    st.set_page_config(page_title="Ongea Labs", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

    _init_state(st)
    inject_css(st, st.session_state.theme, st.session_state.sidebar_open)

    if st.session_state.sidebar_open:
        sidebar_view(st)

    topbar(st)

    chat = _get_active_chat(st)
    render_feed(st, chat)
    input_bar(st)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--lang", type=str, default="swh")
    args = parser.parse_args()

    if args.train:
        launch_training(args.lang)
    else:
        run_app()
