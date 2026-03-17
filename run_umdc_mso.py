#!/usr/bin/env python3
"""
run_umdc_mso.py — run_umdc.py에 --mso_views 옵션을 추가한 래퍼
사용법: python run_umdc_mso.py --mso_views 9 --k_shot 5 ...
"""

import sys
import argparse
import importlib.util
import os

# ── 1) --mso_views 를 먼저 파싱해서 제거
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--mso_views", type=int, default=27, choices=[9, 15, 27])
pre_args, remaining = pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

# ── 2) MSO 패치 적용
from mso_views_patch import patch_mso_views
patch_mso_views(pre_args.mso_views)

# ── 3) run_umdc.py 실행
script_dir = os.path.dirname(os.path.abspath(__file__))
run_umdc_path = os.path.join(script_dir, "run_umdc.py")

spec = importlib.util.spec_from_file_location("run_umdc", run_umdc_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)