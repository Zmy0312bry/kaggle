"""本地快速验证：加载 checkpoint 对 train_soundscapes 推理，看是否输出非零概率。"""
from __future__ import annotations
import sys, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from birdclef2026.model import BirdCLEFModel
from birdclef2026.audio import LogMelExtractor, crop_or_pad, load_audio
from birdclef2026.utils import load_json

CKPT_PATH = "outputs/exp003/fold0_best.pt"
SPECIES_PATH = "data/processed/species_list.json"
AUDIO_GLOB = "data/birdclef-2026/train_soundscapes/*.ogg"

device = torch.device("cpu")

# 加载模型
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
species = load_json(SPECIES_PATH)
model = BirdCLEFModel(
    model_name=ckpt["model_name"],
    num_classes=len(species),
    pretrained=False,
    dropout=ckpt.get("dropout", 0.2),
    pooling=ckpt.get("pooling", "avg"),
    head_hidden=ckpt.get("head_hidden", 0),
    drop_path_rate=ckpt.get("drop_path", 0.0),
).to(device).eval()
model.load_state_dict(ckpt["model"], strict=True)

extractor = LogMelExtractor(sample_rate=32000).to(device).eval()

# 随机找一个 train_soundscapes 文件测试
audio_files = sorted(Path(".").glob(AUDIO_GLOB))
if not audio_files:
    print("找不到 train_soundscapes 文件")
    sys.exit(1)

print(f"测试文件数: {len(audio_files)}")
print(f"模型: {ckpt['model_name']}  pooling={ckpt['pooling']}  species={len(species)}")
print(f"val_auc={ckpt['auc']:.5f}")

for path in audio_files[:3]:
    audio = load_audio(path, sample_rate=32000)
    # 取一个 5 秒窗口
    audio = crop_or_pad(audio[:5*32000], 5*32000, random_crop=False)
    waveform = torch.from_numpy(audio.astype(np.float32)).to(device)
    spec = extractor(waveform).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spec).cpu().numpy()[0]

    probs = 1.0 / (1.0 + np.exp(-logits))
    top5 = np.argsort(probs)[-5:][::-1]

    print(f"\n📁 {path.name}")
    print(f"   logits: min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f}")
    print(f"   probs > 0.5: {(probs > 0.5).sum()}/{len(probs)}")
    print(f"   max_prob: {probs.max():.6f}")
    print(f"   top5: ", end="")
    for idx in top5:
        print(f"{species[idx]}={probs[idx]:.4f} ", end="")
    print()
