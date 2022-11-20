# @Time     :2022/10/12 18:55
# @Author   :LuXin
# @Function :实现4阶Runge-Kutta方法
import numpy as np
import matplotlib.pyplot as plt



y = lambda x:-2-x+np.exp(x) #原函数
fun = lambda x,y:x+y+1      #导数

start = 0
end = 1
step = .1

x_init = 0
y_init = -1
x = np.arange(start,end,step)
print(x)
def runge_kutta(fun,x_init,y_init,step,k):
    y_list = [y_init]
    for i in range(k):
        k1 = step*fun(x_init,y_init)
        k2 = step*fun(x_init+(step/2),y_init+0.5*k1)
        k3 = step*fun(x_init+0.5*step,y_init+0.5*k2)
        k4 = step*fun(x_init+step,y_init+k3)

        y_now = y_init + (1/6)*(k1+2*k2+2*k3+k4)
        y_init = y_now
        x_init += step
        y_list.append(y_now)

    return y_list

y_samples = y(x)
y_predict = runge_kutta(fun,x_init,y_init,step,len(x)-1)
plt.plot(x,y_samples,label='samples')
plt.plot(x,y_predict,label='runge-kutta',ls='--')
plt.legend()
plt.show()