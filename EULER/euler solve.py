# @Time     :2022/9/13 21:15
# @Author   :LuXin
# @Function :利用欧拉法来解微分方程组

import numpy as np
import matplotlib.pyplot as plt

"""
此例用微分法来逼近sin(x),若要逼近其他函数，自行修改f(x)与dev(x)
ELUER法公式
y_new = y_old + y`_old*step
"""

#要逼近的函数
def f(x):
    return np.sin(x)

#函数的导数
def dev(x):
    return np.cos(x)

step = 0.1 #计算步长
start,end = 0,20 #自变量的定义域范围

x = np.arange(start,end,step)
y_real = f(x)
print(x)
y_new = f(start) #初始值
y_old = f(start)

y_predict = []
for n in range(len(x)):
    if n == start:
        y_predict.append(y_new)

    else:
        x_old = x[n-1]
        y_new = y_old + dev(x_old)*step
        y_predict.append(y_new)
        y_old = y_new
y_predict = np.array(y_predict)
mae = abs(y_real-y_predict)/y_real
mae = mae[~np.isnan(mae)]
mae = np.mean(mae)

plt.plot(x,y_predict,label='y_predict')
plt.plot(x,y_real,label='y_real',ls='--')
plt.title(f'Euler Method;step={step};error={mae}')
plt.legend()
plt.show()


