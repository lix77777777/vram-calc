"""实验 3: 框架开销（formulas.md §6；待验证-8）.

CUDA context 不计入 torch 统计, 用 nvidia-smi 的进程占用减 torch reserved 得到.
用法: python exp3_overhead.py
"""

import os
import subprocess

import torch

from utils import fmt, require_cuda, save_result, snapshot


def smi_used_by_me() -> int:
    try:
        out = _smi_query()
    except Exception as e:                       # noqa: BLE001
        print("nvidia-smi 查询失败(不影响其余实验):", e)
        return -1
    me = os.getpid()
    for line in out.strip().splitlines():
        pid, mem = line.split(",")
        if int(pid) == me:
            return int(mem) * 1024 ** 2
    return -1


def _smi_query() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True)


def main() -> None:
    require_cuda()
    torch.ones(1, device="cuda")          # 触发 CUDA context
    s = snapshot()
    smi = smi_used_by_me()
    ctx = smi - s["reserved"] if smi > 0 else -1
    print("torch reserved:", fmt(s["reserved"]), "| nvidia-smi 进程占用:",
          fmt(smi) if smi > 0 else "N/A(容器内可能拿不到)")
    if ctx >= 0:
        print("CUDA context ≈", fmt(ctx))
    save_result("exp3_overhead", {"reserved": s["reserved"],
                                  "smi_used": smi, "cuda_context": ctx})


if __name__ == "__main__":
    main()
