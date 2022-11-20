# @Time     :2022/11/8 10:02
# @Author   :LuXin
# @Function :Tomas追赶发解方程组

"""
原方程
-u^(2)+xu=(x-1)exp
u(0)=1,u(1)=e
u^(2)代表u的二阶导
"""
import matplotlib.pyplot as plt
import numpy as np
from icecream.icecream import ic

start = 0
end = 1
num = 20

step = (end-start)/num
print(step)
x = np.arange(start,end+step,step)
length = len(x)
ic.enable()
#创建解集
u = np.zeros(length)
u[0] = 1
u[-1] = np.e

#求系数a
a = np.ones(length)/(step**2)
ic(a)
#求系数b
b = -2*np.ones(length)/(step**2) + x
ic(b)
#求系数c
c = a
ic(c)

#求系数d
dx = lambda x:(x-1)*np.exp(x)
d = [dx(x_i) for x_i in x]
d[1] = dx(x[1])+u[0]/(step**2)
d[-2] = dx(x[-2])+u[-1]/(step**2)
ic(d)
ic(d[-2])

#求系数e
e = np.zeros(length)
for i in range(1,length,1):
    e[i] = c[i]/(b[i]-a[i]*e[i-1])
ic(e)
#求系数f
f = np.zeros(length)
for i in range(1,length,1):
    f[i] = (d[i]+a[i]*f[i-1])/(b[i]-a[i]*e[i-1])
ic(f)

for i in range(num-1,0,-1):
    print(u[i+1])
    u[i] = e[i] * u[i+1] + f[i]

ic(u)

plt.plot(x,np.exp(x))
plt.scatter(x,u)
plt.show()








