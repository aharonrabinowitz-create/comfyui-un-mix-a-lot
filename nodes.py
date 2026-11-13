"""ComfyUI nodes: split audio into voices, music and effects (BandIt v2), and mix the pieces back together.

BandIt v2 (Watcharasupat et al.): code Apache-2.0 (vendored in ./bandit_v2), weights CC BY-SA 4.0 (Zenodo record
12701995). The weights are downloaded on first use into ComfyUI's models/un_mix_a_lot folder, never bundled.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .chunking import crossfade_add, plan_chunks

try:  # inside ComfyUI
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils as cutils
except Exception:  # standalone (tests)
    folder_paths = mm = cutils = None

SR = 48_000  # BandIt v2 works on mono 48 kHz audio
CKPT_URL = "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt"
CKPT_BYTES = 446_680_129
CKPT_NAME = "bandit-v2-multi.ckpt"
SRC_TO_STEM = {"speech": "voices", "music": "music", "sfx": "effects"}
CHUNK_S, OVERLAP_S, BATCH = 8.0, 1.0, 4
MODEL_KWARGS = dict(
    in_channels=1, band_type="musical", n_bands=64, normalize_channel_independently=False,
    treat_channel_as_feature=True, n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True,
    rnn_type="GRU", mlp_dim=512, hidden_activation="Tanh", hidden_activation_kwargs=None,
    complex_mask=True, use_freq_weights=True, n_fft=2048, win_length=2048, hop_length=512,
    window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True,
    pad_mode="reflect", onesided=True, fs=SR)


# ----------------------------------------------------------------------------- weights
def weights_dir() -> Path:
    env = os.environ.get("UNMIX_MODELS_DIR")
    base = Path(env) if env else (Path(folder_paths.models_dir) / "un_mix_a_lot" if folder_paths
                                  else Path.home() / ".cache" / "un_mix_a_lot")
    base.mkdir(parents=True, exist_ok=True)
    return base


def ckpt_path() -> Path:
    return weights_dir() / CKPT_NAME


def ensure_weights() -> Path:
    p = ckpt_path()
    if p.exists() and p.stat().st_size == CKPT_BYTES:
        return p
    print("[Un-Mix-a-Lot] Downloading BandIt v2 weights (about 447 MB, one time) ...")
    tmp = p.with_suffix(".part")
    last = 0.0
    with urllib.request.urlopen(CKPT_URL, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or CKPT_BYTES)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if time.time() - last > 5:
                print(f"[Un-Mix-a-Lot]   {done / 1e6:.0f} of {total / 1e6:.0f} MB")
                last = time.time()
    if tmp.stat().st_size != CKPT_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The BandIt v2 download was incomplete. Run the node again to retry.")
    tmp.replace(p)
    return p


_MODEL = {"m": None, "dev": None}


def load_model(device: torch.device):
    if _MODEL["m"] is not None:
        if _MODEL["dev"] != str(device):
            _MODEL["m"].to(device)
            _MODEL["dev"] = str(device)
        return _MODEL["m"]
    from .bandit_v2.bandit import Bandit
    path = ensure_weights()
    model = Bandit(stems=list(SRC_TO_STEM), **MODEL_KWARGS)
    # weights_only=True: the file is a downloaded pickle, so never allow arbitrary code to run.
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"BandIt v2 weights don't match the model ({len(missing)} missing)")
    model = model.eval().to(device)
    _MODEL["m"], _MODEL["dev"] = model, str(device)
    return model


def free_model():
    _MODEL["m"] = None
    _MODEL["dev"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You chose cuda, but no NVIDIA GPU is available. Pick auto or cpu.")
        return torch.device("cuda")
    if mm is not None:
        return mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- audio helpers
def _resample(x: torch.Tensor, a: int, b: int) -> torch.Tensor:
    if a == b:
        return x
    import torchaudio.functional as F
    return F.resample(x, a, b)


def _separate_mono(model, device, wav: np.ndarray, pbar=None) -> dict[str, np.ndarray]:
    n = len(wav)
    spans = plan_chunks(n, SR, CHUNK_S, OVERLAP_S) if n > CHUNK_S * SR else [(0, n)]
    ov = int(OVERLAP_S * SR)
    acc = {s: np.zeros(n, np.float32) for s in SRC_TO_STEM}
    for i in range(0, len(spans), BATCH):
        if mm is not None:
            mm.throw_exception_if_processing_interrupted()
        group = spans[i:i + BATCH]
        width = max(e - s for s, e in group)
        batch = np.zeros((len(group), width), np.float32)
        for j, (s, e) in enumerate(group):
            batch[j, :e - s] = wav[s:e]
        x = torch.from_numpy(batch).float().to(device)[:, None, :]
        with torch.inference_mode():
            out = model({"mixture": {"audio": x}})["estimates"]
        est = {k: v["audio"][:, 0, :].float().cpu().numpy() for k, v in out.items()}
        for j, (s, e) in enumerate(group):
            for k in SRC_TO_STEM:
                crossfade_add(acc[k], est[k][j, :e - s], s, first=(i + j == 0),
                              last=(i + j == len(spans) - 1), overlap=ov)
        if pbar is not None:
            pbar.update(len(group))
    return {SRC_TO_STEM[k]: v for k, v in acc.items()}


def _audio(wave: torch.Tensor, sr: int) -> dict:
    return {"waveform": wave, "sample_rate": sr}


# ----------------------------------------------------------------------------- nodes
class UnMixBandIt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            "keep_model_loaded": ("BOOLEAN", {"default": False,
                                              "tooltip": "Keep the model in memory for faster repeat runs."}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("voices", "music", "effects")
    FUNCTION = "run"
    CATEGORY = "audio/Un-Mix-a-Lot"
    DESCRIPTION = ("Splits audio into voices, music and sound effects with BandIt v2. "
                   "The three pieces add back up to roughly the original.")

    def run(self, audio, device, keep_model_loaded):
        wave, sr = audio["waveform"], int(audio["sample_rate"])  # wave: [batch, channels, samples]
        b, c, t = wave.shape
        dev = pick_device(device)
        model = load_model(dev)
        total = 0
        chunks_per = []
        for _ in range(b):
            n48 = int(round(t * SR / sr))
            chunks_per.append(len(plan_chunks(n48, SR, CHUNK_S, OVERLAP_S)) if n48 > CHUNK_S * SR else 1)
        total = sum(chunks_per)
        pbar = cutils.ProgressBar(total) if cutils is not None else None
        outs = {"voices": [], "music": [], "effects": []}
        try:
            for i in range(b):
                mono = wave[i].float().mean(0).cpu()
                mono48 = _resample(mono, sr, SR).numpy().astype(np.float32)
                stems = _separate_mono(model, dev, mono48, pbar)
                for k, v in stems.items():
                    y = _resample(torch.from_numpy(v), SR, sr)
                    y = y[:t] if y.shape[0] >= t else torch.nn.functional.pad(y, (0, t - y.shape[0]))
                    outs[k].append(y[None, :].repeat(c, 1))  # same channel count as the input
        finally:
            if not keep_model_loaded:
                free_model()
        return tuple(_audio(torch.stack(outs[k]).contiguous(), sr) for k in ("voices", "music", "effects"))


class UnMixMix:
    @classmethod
    def INPUT_TYPES(cls):
        g = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {
            "required": {"prevent_clipping": ("BOOLEAN", {"default": True})},
            "optional": {
                "voices": ("AUDIO",), "voices_gain": ("FLOAT", g),
                "music": ("AUDIO",), "music_gain": ("FLOAT", g),
                "effects": ("AUDIO",), "effects_gain": ("FLOAT", g),
            }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/Un-Mix-a-Lot"
    DESCRIPTION = "Mixes voices, music and effects back together. Set a gain to 0 to remove that part."

    def run(self, prevent_clipping, voices=None, voices_gain=1.0, music=None, music_gain=1.0,
            effects=None, effects_gain=1.0):
        parts = [(a, g) for a, g in ((voices, voices_gain), (music, music_gain), (effects, effects_gain))
                 if a is not None]
        if not parts:
            raise ValueError("Connect at least one input (voices, music or effects).")
        sr = int(parts[0][0]["sample_rate"])
        waves = []
        for a, g in parts:
            w = a["waveform"].float()
            if int(a["sample_rate"]) != sr:
                w = _resample(w, int(a["sample_rate"]), sr)
            waves.append(w * g)
        b = max(w.shape[0] for w in waves)
        ch = max(w.shape[1] for w in waves)
        n = max(w.shape[2] for w in waves)
        mix = torch.zeros(b, ch, n)
        for w in waves:
            w = w.expand(b, -1, -1) if w.shape[0] == 1 else w
            w = w.expand(-1, ch, -1) if w.shape[1] == 1 else w
            mix[:w.shape[0], :w.shape[1], :w.shape[2]] += w
        if prevent_clipping:
            peak = mix.abs().amax()
            if peak > 1.0:
                mix = mix / peak
        return (_audio(mix.contiguous(), sr),)


NODE_CLASS_MAPPINGS = {"UnMixBandIt": UnMixBandIt, "UnMixMix": UnMixMix}
NODE_DISPLAY_NAME_MAPPINGS = {"UnMixBandIt": "Un-Mix: Voices / Music / Effects (BandIt v2)",
                              "UnMixMix": "Un-Mix: Mix stems"}
