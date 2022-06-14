import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([2.978, 3.481, 3.548, 3.647, 3.706,
             3.782, 3.867, 3.955, 4.025, 4.083, 4.193])
RMSE = list()
for i in range(1, 13, 1):
    z1 = np.polyfit(x, y, i)  # 多次拟合，8代表次数
    p1 = np.poly1d(z1)  # 拟合多项式
    print(p1)  # 在屏幕上打印拟合多项式
    y_hat = np.polyval(z1, x)
    rmse_t = np.linalg.norm(y-y_hat, ord=2)/(len(y)**0.5)
    RMSE.append(rmse_t)
print("RMSE:", RMSE)
ax = plt.gca()
# 设置坐标间隔 1
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)

plt.barh(range(1, len(RMSE)+1), RMSE)  # 纵轴柱状图

# 标注柱状图值
for i in range(len(RMSE)):
    plt.text(RMSE[i]+0.001, i+1-0.15, round(RMSE[i], 4), ha="left", wrap=True)
plt.xlim(0, 0.13)
plt.xlabel('RMSE')  # x轴名
plt.ylabel('Degree of polynomial')  # y轴名
plt.title('Polynomial degree RMSE comparison')
plt.savefig("D:\Code\生成图片\\拟合曲线RMSE对比.png")
plt.show()
