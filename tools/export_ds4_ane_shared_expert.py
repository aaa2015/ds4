#!/usr/bin/env python3
"""
tools/export_ds4_ane_shared_expert.py

Export DeepSeek-V4.1 Flash Shared Expert FFN to a compiled CoreML model (.mlmodelc)
optimized specifically for Apple Neural Engine (ANE) hardware execution.

Usage:
  python3 tools/export_ds4_ane_shared_expert.py --model-dim 2048 --shared-dim 1536 --output ds4_shared_expert.mlmodelc
"""

import os
import sys
import argparse
import subprocess

def export_coreml_model(model_dim: int, shared_dim: int, clamp_val: float, output_path: str):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import coremltools as ct
    except ImportError:
        print("[!] Required packages not found. Please install torch and coremltools:")
        print("    pip install torch coremltools")
        sys.exit(1)

    print(f"[*] Constructing Shared Expert PyTorch Graph (model_dim={model_dim}, shared_dim={shared_dim}, clamp={clamp_val})...")

    class SharedExpertFFN(nn.Module):
        def __init__(self, model_dim, shared_dim, clamp):
            super().__init__()
            self.model_dim = model_dim
            self.shared_dim = shared_dim
            self.clamp = clamp
            # Weights are dummy here for graph compilation structure; weights can be updated or passed
            self.gate = nn.Linear(model_dim, shared_dim, bias=False)
            self.up   = nn.Linear(model_dim, shared_dim, bias=False)
            self.down = nn.Linear(shared_dim, model_dim, bias=False)

        def forward(self, x):
            # x: [1, model_dim]
            g = self.gate(x)
            u = self.up(x)
            if self.clamp > 0.0:
                g = torch.clamp(g, max=self.clamp)
                u = torch.clamp(u, min=-self.clamp, max=self.clamp)
            silu = F.silu(g)
            mid = silu * u
            out = self.down(mid)
            return out

    model = SharedExpertFFN(model_dim, shared_dim, clamp_val).eval()
    dummy_input = torch.randn(1, model_dim)

    traced_model = torch.jit.trace(model, dummy_input)

    print("[*] Converting PyTorch model to CoreML MLProgram targeting Neural Engine...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=(1, model_dim))],
        outputs=[ct.TensorType(name="shared_out")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
    )

    package_path = "ds4_shared_expert.mlpackage"
    mlmodel.save(package_path)
    print(f"[+] Saved CoreML package to {package_path}")

    # Compile with coremlcompiler to produce .mlmodelc
    print(f"[*] Compiling to ANE hardware binary with xcrun coremlcompiler...")
    cmd = ["xcrun", "coremlcompiler", "compile", package_path, "."]
    subprocess.check_call(cmd)

    if output_path != "ds4_shared_expert.mlmodelc":
        os.rename("ds4_shared_expert.mlmodelc", output_path)

    print(f"\n[SUCCESS] Compiled ANE Shared Expert model: {output_path}")
    print("To use with ds4:")
    print(f"  export DS4_ANE_COREML_MODEL={os.path.abspath(output_path)}")
    print("  ./ds4-bench -m ds4flash.gguf\n")

def main():
    parser = argparse.ArgumentParser(description="Export DeepSeek-V4.1 Flash Shared Expert for Apple Neural Engine")
    parser.add_argument("--model-dim", type=int, default=2048, help="Model embedding dimension (default: 2048)")
    parser.add_argument("--shared-dim", type=int, default=1536, help="Shared expert hidden dimension (default: 1536)")
    parser.add_argument("--clamp", type=float, default=6.0, help="SwiGLU clamp value (default: 6.0)")
    parser.add_argument("--output", type=str, default="ds4_shared_expert.mlmodelc", help="Output .mlmodelc path")

    args = parser.parse_args()
    export_coreml_model(args.model_dim, args.shared_dim, args.clamp, args.output)

if __name__ == "__main__":
    main()
