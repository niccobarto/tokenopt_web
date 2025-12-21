import io
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

import cv2
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

@dataclass
class SRConfig:
    model_name: str = "realesrgan-x4plus"
    scale: int = 4
    weights_path: str = "/workspace/models/realesrgan/realesrgan-x4plus.pth"
    tile: int = 0  # 0 = nessun tiling. Se OOM, imposta 256/512
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = True  # FP16 se GPU lo supporta (quasi sempre su RunPod)
    device: str = "cuda"

_model_lock = threading.Lock()
_model: Optional[RealESRGANer] = None
_model_cfg: Optional[SRConfig] = None

def _build_model(cfg:SRConfig) -> RealESRGANer:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: torch.cuda.is_available() == False")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=cfg.scale,
    )
    if not os.path.exists(cfg.weights_path):
        raise RuntimeError(
            "Pesi Real-ESRGAN non trovati.\n"
            f"Expected: {cfg.weights_path}\n\n"
            "Metti il file .pth nel volume persistente (es. /workspace/models/realesrgan/) "
            "e riprova."
        )

    upsampler = RealESRGANer(
        scale=cfg.scale,
        model_path=cfg.weights_path,
        model=model,
        tile=cfg.tile,
        tile_pad=cfg.tile_pad,
        pre_pad=cfg.pre_pad,
        half=cfg.half,
        device=torch.device(cfg.device),
    )
    return upsampler

def get_upsampler(cfg:SRConfig) -> RealESRGANer:
    global _model, _model_cfg
    with _model_lock:
        if _model is None or _model_cfg != cfg:
            _model = _build_model(cfg)
            _model_cfg = cfg
    return _model

#--------------------------
# API: bytes -> bytes
#--------------------------

def run_realesgan(input_bytes: bytes, cfg: Optional[SRConfig] = None) -> bytes:

    if cfg is None:
        cfg = SRConfig()

    try:
        img = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError("Input non è un'immagine valida (bytes non decodificabili).") from e

    rgb=np.array(img, dtype=np.uint8)
    bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    upsampler = get_upsampler(cfg)

    try:
        out_bgr, _ = upsampler.enhance(bgr, outscale=cfg.scale)
    except RuntimeError as e:
        # Tipico: OOM GPU. Suggerisci tile.
        raise RuntimeError(
            "Errore durante SR (possibile OOM GPU). "
            "Prova a impostare SRConfig.tile=256 o 512."
        ) from e
    except Exception as e:
        raise RuntimeError("Errore inatteso durante Real-ESRGAN enhance().") from e

    # BGR -> RGB -> PNG bytes
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(out_rgb)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()