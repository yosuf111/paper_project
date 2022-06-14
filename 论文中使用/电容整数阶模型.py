import numpy as np
import matplotlib.pyplot as plt
R_0 = 0.002  # mΩ
R_1 = 0.001  # mΩ
C_0 = 20000  # F
C_1 = 150000  # F


def my_func(R_0, R_1, C_0, C_1, m, n):
    w = np.linspace(0.0002, 2.78, 10000)
    Z_0 = R_0+1/((1j*w)**m*C_0)+1/(1/R_1+(1j*w)**n*C_1)
    Z_RE = np.real(Z_0)
    Z_IM = -np.imag(Z_0)
    return Z_RE, Z_IM


Z_RE, Z_IM = my_func(R_0, R_1, C_0, C_1, 1, 1)
Z_RE_f, Z_IM_f = my_func(R_0, R_1, C_0, C_1, 0.99, 0.98)
plt.plot(Z_RE, Z_IM, 'r', label='Integer Order')
plt.plot(Z_RE_f, Z_IM_f, 'b', label='Fractional Order')
# 横轴坐标轴命名
plt.xlabel('Z_RE')
#垂直轴命名
plt.ylabel('|Z_IM|')
# 标注图像高频区间
plt.text(0.002, 0.01, 'High Frequency Region', ha="left", wrap=True)
# 标注图像低频区间
plt.text(0.003, 0.045, 'Low Frequency Region', ha="left", wrap=True)
plt.grid('True')
plt.legend(loc='best')
plt.title('Nyquist Diagram')
plt.show()
