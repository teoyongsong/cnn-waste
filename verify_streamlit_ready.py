#!/usr/bin/env python3
"""Pre-flight checks before Streamlit deployment. Run from cnn/: python verify_streamlit_ready.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    print(f"  !!  {msg}")


def main() -> int:
    print(f"Checking deployment readiness under: {APP_DIR}\n")
    exit_code = 0

    # Required static files
    for name, desc in [
        ("class_names.json", "30 class labels"),
        ("deployment_config.json", "architecture + checkpoint name"),
        ("streamlit_app.py", "Streamlit entrypoint"),
        ("waste_model_loader.py", "model loader"),
    ]:
        p = APP_DIR / name
        if p.exists():
            ok(f"{desc}: {name}")
        else:
            fail(f"Missing {name}")
            exit_code = 1

    # class_names count
    classes_path = APP_DIR / "class_names.json"
    if classes_path.exists():
        with open(classes_path, encoding="utf-8") as f:
            classes = json.load(f)
        n = len(classes) if isinstance(classes, list) else 0
        if n == 30:
            ok(f"class_names.json has 30 classes")
        else:
            fail(f"class_names.json expected 30 classes, got {n}")
            exit_code = 1

    # deployment_config
    cfg_path = APP_DIR / "deployment_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("architecture", "")
        ckpt = cfg.get("checkpoint", "")
        if arch and ckpt:
            ok(f"deployment_config: architecture={arch!r}, checkpoint={ckpt!r}")
        else:
            fail("deployment_config.json must set architecture and checkpoint")
            exit_code = 1

    # Checkpoint (warning only — often added after first deploy)
    ckpt_path = APP_DIR / "best_waste_model.pth"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg_ckpt = json.load(f).get("checkpoint", "best_waste_model.pth")
        ckpt_path = APP_DIR / cfg_ckpt
    if ckpt_path.exists():
        mb = ckpt_path.stat().st_size / (1024 * 1024)
        ok(f"Checkpoint found: {ckpt_path.name} (~{mb:.1f} MB)")
    else:
        print(f"  WARN  No checkpoint at {ckpt_path} — train, save .pth, commit or use Cloud secrets.")

    # Import smoke test (needs PyTorch in the active environment)
    try:
        from waste_model_loader import load_class_names, load_deployment_config

        load_class_names()
        load_deployment_config()
        ok("waste_model_loader imports and load_class_names() works")
    except ImportError as e:
        print(f"  WARN  Import test skipped (use env with torch+torchvision): {e}")
    except Exception as e:
        fail(f"Import test failed: {e}")
        exit_code = 1

    print()
    if exit_code == 0:
        print("All structural checks passed. Deploy with:")
        print("  streamlit run streamlit_app.py")
    else:
        print("Fix issues above before deploying.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
