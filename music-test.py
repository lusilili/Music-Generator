import numpy as np
import matplotlib.pyplot as plt

# 定义音频长度范围（秒）
lengths = np.linspace(10, 300, 100)

# 定义生成时间
# 自回归方法：指数关系
autoregressive_time = 2 + 0.01 * lengths ** 1.8
# 序列生成方法：次线性增长
sequence_time = 2 + 0.005 * lengths * np.log(lengths)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(lengths, autoregressive_time, label="Autoregressive Method (MusicGen)", linestyle='--', linewidth=2)
plt.plot(lengths, sequence_time, label="Transformer Sequence Generation", linestyle='-', linewidth=2)

# 图表美化
plt.title("Comparison of Music Generation Methods", fontsize=16)
plt.xlabel("Generated Music Length (seconds)", fontsize=14)
plt.ylabel("Generation Time (seconds)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
