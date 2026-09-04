"""스모크 테스트 — 가중치/데이터 없이도 통과해야 하는 최소 검증.

1) RRDBNetA2V5B 인스턴스화 + 순전파 (같은 크기 출력, [0,1])
2) CycleGAN Generator 인스턴스화 + 순전파 (Tanh)
3) SC Loss(VGG 백엔드) 순전파 + backward
4) config 로드
실행: (프로젝트 루트에서) python scripts/smoke_test.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[smoke] device={device}, torch={torch.__version__}")
ok = True


def check(name, fn):
    global ok
    try:
        fn()
        print(f"  ✅ {name}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ❌ {name}: {type(e).__name__}: {e}")


def t_rrdb():
    from src.arch_rrdb_a2v5b import RRDBNetA2V5B

    m = RRDBNetA2V5B().to(device).eval()
    x = torch.rand(1, 3, 64, 64, device=device)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape, f"shape {y.shape}"
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0, "clamp"


def t_cyclegan():
    from cycleGen_model import CycleGANGenerator

    g = CycleGANGenerator(3, 3, 9).to(device).eval()
    x = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    with torch.no_grad():
        y = g(x)
    assert y.shape == x.shape, f"shape {y.shape}"


def t_sc_loss():
    from src.sc_loss import StructuralConsistencyLoss

    # smoke는 VGG 가중치 다운로드 없이 그래프만 검증 (pretrained=False)
    sc = StructuralConsistencyLoss(backend="vgg", device=device, pretrained=False)
    a = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
    b = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
    loss = sc(a, b)
    loss.backward()
    assert loss.item() >= 0


def t_config():
    import yaml

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/paths.yaml"), encoding="utf-8"))
    assert "weights" in cfg and "eval" in cfg


check("RRDBNetA2V5B forward", t_rrdb)
check("CycleGANGenerator forward", t_cyclegan)
check("StructuralConsistencyLoss(vgg) backward", t_sc_loss)
check("config load", t_config)

print("[smoke]", "ALL PASS ✅" if ok else "FAILURES ❌")
sys.exit(0 if ok else 1)
