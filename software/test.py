import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
# famp值 (K)
famp = np.array([
    13.58, 14.64, 15.67, 16.74, 17.78, 20.97, 
    23.10, 33.85, 76.98, 120.24, 228.04, 462.89
])

# 阻抗值 (Kohm)
impedance = np.array([
    1, 2, 3, 4, 5, 8, 
    10, 20, 60, 100, 200, 500
])

# ==========================================
# 2. 绘图风格设置
# ==========================================
# 设置字体，按顺序尝试，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 颜色设置
matlab_red = '#A2142F'

# 创建画布
plt.figure(figsize=(8, 5), dpi=150)

# ==========================================
# 3. 绘制曲线
# ==========================================
# 绘制连线和数据点
plt.plot(famp, impedance, 
         color=matlab_red,      
         linestyle='-',          # 实线
         linewidth=1.5,          
         marker='o',             # 圆点标记
         markersize=5,           
         label='阻抗标定数据')

# ==========================================
# 4. 细节调整
# ==========================================
# 开启网格
plt.grid(True, which='major', linestyle='--', alpha=0.6)

# 设置坐标轴范围
plt.xlim(0, 500)
plt.ylim(0, 600)

# 设置标签和标题
plt.xlabel('famp值 (K)', fontsize=12)
plt.ylabel('阻抗 (Kohm)', fontsize=12)
plt.title('famp值 - 阻抗标定曲线', fontsize=14)

# 设置刻度密度
plt.xticks(np.arange(0, 501, 50))
plt.yticks(np.arange(0, 601, 50))

# 添加图例
plt.legend(loc='upper left')

# 紧凑布局
plt.tight_layout()

# 显示图表
print("正在生成图表窗口...")
plt.show()