import matplotlib.pyplot as plt
import numpy as np
from icecream.icecream import ic

class ODE_BVP_SOLVE():
    """
    求解ODE-BVP方程
    -u'' + xu = (x-1)e^x
    u(0)=1，u(1)=e
    真解为e^x
    """
    def __init__(self):
        """
        生成样本点
        生成真解
        生成右端函数
        """
        ic.disable()
        start = 0 #样本点左端点
        end = 1 #样本点右端点
        N = 30 #节点数（包含端点和内点）
        h = (end-start)/(N-1) #N个数，N-1个间隔，这样计算的是每一个间隔的长度，也就是步长
        self.h = h

        """
        端点值
        段点值是由真解确定的，对于不同的问题，真解不同，段点值也就不同
        """
        u0 = 1
        uN = np.e
        self.u0,self.uN = u0,uN

        """
        生成所有节点
        注意，所有节点中，两端点的解已知，需要求解的是除两端点以外的点
        """
        x_all = np.linspace(start,end,N) #所有节点
        x_solve = x_all[1:-1] #求解节点
        self.x_all = x_all
        self.x_solve = x_solve

        #该问题的真解是e^x,这里计算出对应的真解值
        u_real = np.exp(x_all)
        self.u_real = u_real
        ic(x_all)
        ic(x_solve)

        # 右端函数
        f = lambda x: (x - 1) * np.exp(x)
        self.f = f

    def coefficient_generate(self):
        """
        将追赶法中涉及到的系数全部计算出来
        返回这些系数作为中间过程检查
        如果要求解的问题改变，在这里修改系数的计算方式即可
        :return: a,b,c,d,e,f
        """
        # 系数a
        a = np.ones(len(self.x_solve)) / (self.h ** 2)
        ic(a)

        # 系数b
        b = 2 / np.ones(len(self.x_solve)) / (self.h**2) + self.x_solve
        ic(b)

        # 系数c
        c = a
        ic(c)

        # 系数d
        d = np.array([self.f(x) for x in self.x_solve])
        d[0] = d[0] + self.u0 / (self.h ** 2)
        d[-1] = d[-1] + self.uN / (self.h ** 2)
        ic(d)

        # 系数e
        e = np.zeros(len(self.x_solve))
        e[0] = c[0] / b[0]
        for i in range(1, e.shape[0], 1):
            e[i] = c[i] / (b[i] - a[i] * e[i - 1])
        ic(e)

        # 系数f
        #注意区分这里的系数f和右端函数的f
        f = np.zeros(len(self.x_solve))
        f[0] = d[0] / b[0]
        for i in range(1, f.shape[0], 1):
            f[i] = (d[i] + a[i] * f[i - 1]) / (b[i] - a[i] * e[i - 1])
        ic(f)

        return a,b,c,d,e,f


    def solve(self,e,f):
        """
        :param e: 系数e
        :param f: 系数f
        :return:
        """
        """
        U_{N-1} = e_{N-1}*U_{N} + f_{N-1} = f_{N-1}
        注意，这里的U_{N} = 0，这里的U_{N}并不是最右段点的已知值，而是0
        因为这里的U_{N}代表的是稀疏矩阵形式里的那个U_{N},
        在实际求解时，边界点已知，求出值后移到右端，作为系数了，所以这时认为U_{0}和U_{N}是0
        所以最后一个求解节点的解直接等于f_{N-1}
        """
        u = np.zeros(len(self.x_solve))
        u[-1] = f[-1]

        #最后一个点不算，一直算到最左端
        for i in range(-2,-len(self.x_solve)-1,-1):
            u[i] = e[i]*u[i+1] + f[i]

        #解集中不包含断点，为了作图直观，这里加入断点值
        u = np.insert(u,0,1)
        u = np.append(u,self.uN)
        self.u = u
        ic(u)

        return u

    def result_plot(self):
        plt.scatter(self.x_all,self.u,label='solve',c='#000075',marker='*')
        plt.plot(self.x_all,self.u_real,label='sample',c='#800000')
        plt.xlabel('X')
        plt.ylabel('Solve')
        plt.legend()
        plt.show()

calculator = ODE_BVP_SOLVE()
a,b,c,d,e,f = calculator.coefficient_generate()
calculator.solve(e,f)
calculator.result_plot()