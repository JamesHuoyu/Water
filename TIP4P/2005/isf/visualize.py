import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

# 读取 CSV 文件
filename = "Fs_Fk_k2.00_dk0.15.csv"  # 替换为你实际的文件名
with open(filename, "r") as f:
    lines = f.readlines()

# 解析元数据
meta_line = [line for line in lines if line.startswith("# {")][0]
meta = json.loads(meta_line[2:])
tau_alpha = meta.get("tau_alpha_ps_e1", None)
logtime = meta.get("logtime", False)

# 读取数据部分
df = pd.read_csv(filename, comment="#")

# 绘图
plt.figure(figsize=(7, 5))
x = df["t_ps"]
y = df["Fs"]
yerr = df["sem"]

if logtime:
    plt.xscale("log")

plt.errorbar(x, y, yerr=yerr, fmt="o", markersize=4, label="Fs(k,t)", capsize=3)

# 标记 tau_alpha
if tau_alpha is not None:
    plt.axvline(x=tau_alpha, color="r", linestyle="--", label=r"$\tau_\alpha$")
    plt.text(tau_alpha, 0.5, r"$\tau_\alpha$", color="r")

# 设置图形标签
plt.xlabel("Time (ps)")
plt.ylabel(r"$F_s(k,t)$")
plt.title("Self-intermediate Scattering Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fs_Fk_k2.00_dk0.15.png", dpi=300)
