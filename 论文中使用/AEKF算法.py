from pyexpat.errors import XML_ERROR_XML_DECL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### SOC_OCV函数过程 ###
SOC = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
U_ocv = np.array([2.978, 3.481, 3.548, 3.647, 3.706,
                  3.782, 3.867, 3.955, 4.025, 4.083, 4.193])
matrix_SOC_OCV = np.polyfit(SOC, U_ocv, 8)  # 多次拟合，8代表次数
SOC_OCV = np.poly1d(matrix_SOC_OCV)  # 拟合多项式
dSOC_OCV = SOC_OCV.deriv()
#######################
# 导入测量ut实际数据
data_mea = pd.read_excel('D:\\Code\\数据集\\25摄氏度恒流放电_3.xlsx')
ut_mea = list(data_mea['mV']/1000)
data_length = len(ut_mea)+10000
########################
# 导入电池参数
# 设置电池的估算初始状态
x_hat = np.zeros([data_length, 2, 1])
x_hat[0] = np.array([[0], [1]])
x_check = np.zeros([data_length, 2, 1])
x_check[0] = x_hat[0]
y = np.zeros([data_length, 1])
# y[0] = ut_mea[0]
y[0] = 4.193
########################
## 状态估计参数设置 ##
P_hat = np.zeros([data_length, 2, 2])
P_hat[0] = np.diag([0.01, 0.1])  # 初始化方差矩阵
Q = np.zeros([data_length, 2, 2])
Q[0] = np.diag([0.1, 0.1])  # 过程噪声方差矩阵
R = np.zeros([data_length, 1])
R[0] = np.diag([0.1])  # 测量方差矩阵
P_check = np.zeros([data_length, 2, 2])
K = np.zeros([data_length, 1, 2])
######################
i = -0.1  # 充电为正，放电为负 A
R_d = 0.00057
C_d = 11043.04
tau = R_d*C_d
R_in = 0.0023
eta = 0.9
delt_t = 1


def mat_func(delt_t, tau, R_d, eta):  # A,B矩阵
    # 将z带入关系方程，传出tau和R_d和
    # 目前假设tau为常数，R_d为常数，R_in为常数，其中这些数与z有关
    C_max = 7000
    A_k0 = np.array([[np.exp(-delt_t/tau), 0], [0, 1]])
    B_k0 = np.array([(1-np.exp(-delt_t/tau))*R_d,
                    eta*delt_t/C_max]).reshape(2, 1)
    return A_k0, B_k0


def mat2_func(dSOC_OCV_z, OCV, U_d, R_in, u_k0):  # C,D矩阵
    C_k1 = np.array([-1, dSOC_OCV_z])
    D_k1 = OCV + U_d + R_in*u_k0 - C_k1@x_hat[k+1]
    return C_k1, D_k1


s = []
s.append(y[0])
sd = []
sd.append(y[0]-ut_mea[0])
for k in range(77970):
    A_k0, B_k0 = mat_func(delt_t=delt_t, tau=tau, R_d=R_d, eta=eta)
    x_hat[k+1] = (A_k0@x_hat[k] + B_k0*i)  # 关于x的估计值为 x_^-[k+1]
    z = float(x_hat[k+1][1])
    U_d = float(x_hat[k+1][0])
    C_k1, D_k1 = mat2_func(dSOC_OCV(z), SOC_OCV(z), U_d, R_in, i)
    y[k+1] = C_k1@x_hat[k+1] + D_k1  # 关于u的估计值为 u_^-[k+1]
    P_check[k+1] = A_k0@P_hat[k]@A_k0.T + Q[k]  # 关于P的估计值为 P_^-[k+1]
    x_check[k+1] = A_k0@x_hat[k]+B_k0*i  # 关于x的估计值为 x_^-[k+1]
    if ((C_k1@P_check[k+1]@C_k1.T) + R[k+1]) == 0:
        K[k+1] = K[k]
    else:
        K[k+1] = P_check[k+1]@C_k1.T / \
            ((C_k1@P_check[k+1]@C_k1.T) + R[k+1])  # 关于K的估计值为 K_^-[k+1]
    P_hat[k+1] = (np.eye(2) - K[k+1]@C_k1)@P_check[k+1]  # 关于P的估计值为 P_^-[k+1]

    x_hat[k+1] = x_check[k+1] + K[k+1].T * \
        (y[k+1] - (C_k1@x_check[k+1]+D_k1))  # 关于x的估计值为 x_^-[k+1]
    s.append(y[k+1])
    sd.append(y[k+1]-ut_mea[k+1])


plt.plot(s, 'g', label='s')
plt.plot(ut_mea, 'b', label='ut_mea')
plt.plot(sd, 'r', label='sd')
plt.legend('best')
plt.show()
