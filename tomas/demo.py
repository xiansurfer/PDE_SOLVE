# @Time     :2022/11/8 15:53
# @Author   :LuXin
# @Function :
import numpy as np

#边界条件
u0 = 1
uN = np.exp(1)

N=20
h=1/N
x_all = np.arange(0,1+h,h)
x = x_all[1:-1]
print(len(x))
#右端
fx = lambda x:(x-1)*np.exp(x)
d = fx(x)
d[0] = d[0] + u0/(h**2)
d[-1] = d[-1] +uN/(h**2)

#求稀疏矩阵A
qx = lambda x:x
q = qx(x)
a = np.ones(N-1)/(h**2)
b = 2*np.ones(N-1)/(h**2) + q
c=a


def thomas(a,b,c,d):
    M = len(a)
    u = np.zeros(M)
    e = np.zeros(M)
    f = np.zeros(M)
    e[0] = c[0]/b[0]
    f[0] = d[0]/b[0]

    for i in range(2,M):
        e[i] = c[i]/(b[i]-a[i]*e[i-1])
        f[i] = (d[i]+a[i]*f[i-1]/(b[i]-a[i]*e[i-1]))
    print(u)
    print(f)
    u[M-1] = f[M-1]
    for i in range(M-2,1,-1):
        u[i] = f[i] + e[i]*u[i+1]
    return u
#求解
u = thomas(a,b,c,d)
print(u)
