"""
mso_views_patch.py
------------------
run_umdc.py 에 --mso_views 옵션을 추가하는 패치.

사용법:
    python run_umdc.py --mso_views 9 [나머지 옵션들...]
    → 이 파일을 run_umdc.py 와 같은 디렉토리에 두면 자동 적용.

작동 원리:
    buffer에 저장된 feature가
    - Case A: [N, n_views, D] 형태면 → 앞 mso_views개만 사용 후 평균
    - Case B: 이미 평균된 [N, D] 형태면 → buffer 재로드 불가이므로 
              feature extraction 단계에서 offset subset 사용

    MVREC의 MSO는 offsets 리스트를 순회하며 feature를 합산/평균하므로
    offsets 리스트를 잘라내는 방식으로 구현.
"""

import torch
import numpy as np

# ── offset 정의 (MVREC 원본과 동일한 순서) ──────────────────────────────────
# 3×3 grid × 3 scale = 27
# scale별로 9개씩: scale0(center 영역), scale1(확장), scale2(더 확장)
MSO_OFFSETS_27 = [
    # scale 0 — 9개
    (0,   0,   1.0), (-0.1, 0,   1.0), (0.1,  0,   1.0),
    (0,  -0.1, 1.0), (0,    0.1, 1.0), (-0.1,-0.1, 1.0),
    (0.1,-0.1, 1.0), (-0.1, 0.1, 1.0), (0.1,  0.1, 1.0),
    # scale 1 — 9개
    (0,   0,   1.2), (-0.1, 0,   1.2), (0.1,  0,   1.2),
    (0,  -0.1, 1.2), (0,    0.1, 1.2), (-0.1,-0.1, 1.2),
    (0.1,-0.1, 1.2), (-0.1, 0.1, 1.2), (0.1,  0.1, 1.2),
    # scale 2 — 9개
    (0,   0,   1.5), (-0.1, 0,   1.5), (0.1,  0,   1.5),
    (0,  -0.1, 1.5), (0,    0.1, 1.5), (-0.1,-0.1, 1.5),
    (0.1,-0.1, 1.5), (-0.1, 0.1, 1.5), (0.1,  0.1, 1.5),
]

VIEW_PRESETS = {
    9:  MSO_OFFSETS_27[:9],   # scale 0 only (center crop 9 positions)
    15: MSO_OFFSETS_27[:15],  # scale 0 + scale 1 (앞 15개)
    27: MSO_OFFSETS_27,       # 전체
}


def patch_mso_views(n_views: int):
    """
    fewshot_process.py 또는 modules/ 내의 feature extraction 함수에
    offset subset을 monkey-patch 한다.

    n_views ∈ {9, 15, 27}
    """
    if n_views == 27:
        return  # 기본값, 패치 불필요

    assert n_views in VIEW_PRESETS, f"--mso_views must be one of {list(VIEW_PRESETS)}"
    target_offsets = VIEW_PRESETS[n_views]

    # ── 1) lyus / modules 내 offset 상수 패치 ────────────────────────────────
    patched = False
    for mod_name in ["lyus.mso", "modules.mso", "lyus.feature_extractor",
                     "modules.feature_extractor", "fewshot_process"]:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            for attr in dir(mod):
                val = getattr(mod, attr)
                # offset 리스트로 보이는 속성 찾기
                if isinstance(val, list) and len(val) == 27:
                    setattr(mod, attr, target_offsets)
                    print(f"[mso_patch] Patched {mod_name}.{attr}: 27 → {n_views} views")
                    patched = True
                elif isinstance(val, (torch.Tensor, np.ndarray)) and len(val) == 27:
                    if isinstance(val, torch.Tensor):
                        setattr(mod, attr, val[:n_views])
                    else:
                        setattr(mod, attr, val[:n_views])
                    print(f"[mso_patch] Patched {mod_name}.{attr} tensor: 27 → {n_views} views")
                    patched = True
        except (ImportError, AttributeError):
            continue

    # ── 2) buffer feature가 [N, 27, D] 형태로 저장된 경우 ───────────────────
    # torch.load를 감싸서 로드된 텐서를 슬라이싱
    _orig_load = torch.load

    def _patched_load(f, *args, **kwargs):
        data = _orig_load(f, *args, **kwargs)
        if isinstance(data, torch.Tensor) and data.dim() == 3 and data.shape[1] == 27:
            data = data[:, :n_views, :].mean(dim=1)
            print(f"[mso_patch] Sliced buffer tensor [N,27,D] → [N,{n_views},D].mean(1)")
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[1] == 27:
                    data[k] = v[:, :n_views, :].mean(dim=1)
                    print(f"[mso_patch] Sliced buffer dict['{k}'] [N,27,D] → mean over {n_views} views")
        return data

    torch.load = _patched_load
    print(f"[mso_patch] torch.load patched for {n_views}-view subsampling")

    if not patched:
        print("[mso_patch] WARNING: offset list not found in modules — "
              "buffer slicing only (works if buffer stores [N,27,D])")


# ── run_umdc.py 에 --mso_views 인수 추가 패치 ─────────────────────────────────
def inject_argparse(parser):
    """run_umdc.py의 ArgumentParser에 --mso_views 추가"""
    parser.add_argument(
        "--mso_views", type=int, default=27, choices=[9, 15, 27],
        help="Number of MSO views to use (9/15/27). Default: 27 (all views)."
    )
    return parser


if __name__ == "__main__":
    # 단독 실행 시 동작 확인
    print("MSO offsets:")
    for n in [9, 15, 27]:
        print(f"  {n} views: {len(VIEW_PRESETS[n])} offsets")
