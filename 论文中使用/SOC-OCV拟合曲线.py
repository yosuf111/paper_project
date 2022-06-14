import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([2.978, 3.481, 3.548, 3.647, 3.706,
             3.782, 3.867, 3.955, 4.025, 4.083, 4.193])
z1 = np.polyfit(x, y, 8)  # 多次拟合，8代表次数
print('z1:',z1)
p1 = np.poly1d(z1)  # 拟合多项式
print(p1)  # 在屏幕上打印拟合多项式
dp1s = p1.deriv()
print(dp1s)  # 打印OCV的导数
t = np.array([n for n in np.arange(0, 1, 0.001)])  # 拟合曲线使用，使图更顺滑采样点增多。
yvals = np.polyval(z1, t)  # 也可以使用yvals=np.polyval(z1,x)
y_hat = np.polyval(z1, x)
RMSE = np.linalg.norm(y-y_hat, ord=2)/(len(y)**0.5)
print("RMSE:", RMSE)
plot1 = plt.plot(x, y, '*', label='Measured OCV')
plot2 = plt.plot(t, yvals, 'r', label='fitted OCV')
plt.xlabel('SoC')  # x轴名
plt.ylabel('Voltage(V)')  # y轴名
plt.legend(loc='best')  # 指定legend的位置,读者可以自己help它的用法 图示说明位置
plt.title('Fitted OCV-SOC curve')
plt.savefig('D:\Code\生成图片\\SOC-OCV拟合曲线.png', dpi=600, format='eps')
plt.show()
