# @Time     :2022/10/19 15:59
# @Author   :LuXin
# @Function :实现若干个显式adams方法的求解
import numpy as np
import matplotlib.pyplot as plt

class Adams_method():

    def __init__(self,y,fun,start,end,step):
        """
        生成样本点，并将各种条件赋值给类属性
        :param y:原函数
        :param fun:导数
        :param start:样本点起点
        :param end:样本点终点
        :param step:样本点步长
        """
        # 生成样本点
        self.x_samples = np.arange(start, end, step)  # 节点是已知量
        # 函数精确值是未知量，用来做误差计算和对比
        self.y_samples = y(self.x_samples)

        self.fun = fun
        self.step = step

        #计算日志，用来存放不同方法的计算结果
        self.calculate_log = {}

    def euler(self,x_old,y_old):
        """
        #欧拉法计算下一个节点的函数值
        :param x_old: 上一个节点的值
        :param y_old: 上一个节点的函数值
        :return: 求解节点的函数值
        """
        return y_old + self.step * self.fun(x_old,y_old)

    def explicit_1k_1p(self):
        # 生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[0]: self.y_samples[0]}


        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            y_old = list(self.solve_dic.values())[-1]
            x_old = list(self.solve_dic.keys())[-1]
            x_new = x_old + step
            f_old = fun(x_old,y_old)

            y_new = y_old + self.step*f_old
            self.solve_dic[x_new] = y_new

        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Explicit 1_Step 1_Order'] = \
            {'solve':solve,'error':error}
        return self.solve_dic

    def explicit_2k_2p(self):
        #生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[0]: self.y_samples[0]}

        self.euler_predict(k=2)

        #样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)


        for i in range(calculate_times):
            y_old = list(self.solve_dic.values())[-1]
            x_old = list(self.solve_dic.keys())[-1]
            x_new = x_old + step

            # 计算下一个节点函数值y_{n+2}
            x_n_1 = list(self.solve_dic.keys())[-1]  # x_{n+1} 往前数一个节点
            y_n_1 = list(self.solve_dic.values())[-1]  # y_{n+1} 往前数一个节点的函数值
            f_n_1 = fun(x_n_1, y_n_1)  # f_{n+1} 往前数一个节点的导数值

            x_n = list(self.solve_dic.keys())[-2]  # x_{n} 往前数两个节点
            y_n = list(self.solve_dic.values())[-2]  # y_{n} 往前数两个节点的函数值
            f_n = self.fun(x_n, y_n)  # f_{n+1} 往前数两个节点的导数值

            y_new = y_old + (self.step / 2) * (3 * f_n_1 - f_n)
            self.solve_dic[x_new] = y_new

        else:
            error = self.erro_calculate()
            solve = list(self.solve_dic.values())
            self.calculate_log['Explicit 2_Step 2_Order'] = \
                {'solve': solve, 'error': error}
            return self.solve_dic

    def explicit_3k_3p(self):
        # 生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[0]: self.y_samples[0]}

        # 欧拉法预估，使解集满足要求
        self.euler_predict(k=3)

        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            y_old = list(self.solve_dic.values())[-1]
            x_old = list(self.solve_dic.keys())[-1]
            x_new = x_old + step

            # 计算下一个节点函数值y_{n+3}
            x_n_2 = list(self.solve_dic.keys())[-1]  #x_{n+2} 往前数一个节点
            y_n_2 = list(self.solve_dic.values())[-1]  #y_{n+2} 往前数一个节点的函数值
            f_n_2 = fun(x_n_2, y_n_2)  # f_{n+2} 往前数一个节点的导数值

            x_n_1 = list(self.solve_dic.keys())[-2]  # x_{n+1} 往前数两个节点
            y_n_1 = list(self.solve_dic.values())[-2]  # y_{n+1} 往前数两个节点的函数值
            f_n_1 = self.fun(x_n_1, y_n_1)  # f_{n+1} 往前数两个节点的导数值

            x_n = list(self.solve_dic.keys())[-3]  #x_{n}往前数三个节点
            y_n = list(self.solve_dic.values())[-3]  #y_{n} 往前数三个节点的函数值
            f_n = self.fun(x_n, y_n)  #f_{n} 往前数三个节点的导数值

            y_new = y_old+(self.step/12)*(23*f_n_2-16*f_n_1+5*f_n)
            self.solve_dic[x_new] = y_new
        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Explicit 3_Step 3_Order'] = \
            {'solve': solve, 'error': error}
        return self.solve_dic

    def explicit_4k_4p(self):
        # 生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[0]: self.y_samples[0]}

        # 欧拉法预估，使解集满足要求
        self.euler_predict(k=4)

        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            y_old = list(self.solve_dic.values())[-1]
            x_old = list(self.solve_dic.keys())[-1]
            x_new = x_old + step

            # 计算下一个节点函数值y_{n+4}
            x_n_3 = list(self.solve_dic.keys())[-1]  #x_{n+3} 往前数一个节点
            y_n_3 = list(self.solve_dic.values())[-1]  #y_{n+3} 往前数一个节点的函数值
            f_n_3 = fun(x_n_3, y_n_3)  # f_{n+3} 往前数一个节点的导数值

            x_n_2 = list(self.solve_dic.keys())[-2]  # x_{n+2} 往前数两个节点
            y_n_2 = list(self.solve_dic.values())[-2]  # y_{n+2} 往前数两个节点的函数值
            f_n_2 = self.fun(x_n_2, y_n_2)  # f_{n+2} 往前数两个节点的导数值

            x_n_1 = list(self.solve_dic.keys())[-3]  # x_{n+1} 往前数三个节点
            y_n_1 = list(self.solve_dic.values())[-3]  # y_{n+1} 往前数三个节点的函数值
            f_n_1 = self.fun(x_n_1, y_n_1)  # f_{n+1} 往前数三个节点的导数值

            x_n = list(self.solve_dic.keys())[-4]  #x_{n}往前数四个节点
            y_n = list(self.solve_dic.values())[-4]  #y_{n} 往前数四个节点的函数值
            f_n = self.fun(x_n, y_n)  #f_{n} 往前数四个节点的导数值

            y_new = y_old + (self.step/24)*(55*f_n_3-59*f_n_2+37*f_n_1-9*f_n)
            self.solve_dic[x_new] = y_new
        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Explicit 4_Step 4_Order'] = \
            {'solve': solve, 'error': error}
        return self.solve_dic

    def implicit_1k_2p(self):
        """
        显示2步2阶预估，隐式1步2阶法做校正
        :return: self.solve_dic 解集字典
        """
        # 生成解字典，初始值是已知量，即n点有解
        self.solve_dic = {self.x_samples[i]: self.y_samples[i] for i in range(2)}


        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            "==========显示2步2阶法预测n+2点函数值=========="
            # 欧拉法预测n+1点的解，给显示2步法预测n+2点提供信息
            self.euler_predict(k=2)

            # 计算下一个节点函数值y_{n+2}
            x_n_1 = list(self.solve_dic.keys())[-1]# x_{n+1} 往前数一个节点
            y_n_1 = list(self.solve_dic.values())[-1]#y_{n+1} 往前数一个节点的函数值
            f_n_1 = fun(x_n_1, y_n_1)  # f_{n+1} 往前数一个节点的导数值

            x_n = list(self.solve_dic.keys())[-2]  # x_{n} 往前数两个节点
            y_n = list(self.solve_dic.values())[-2]  # y_{n} 往前数两个节点的函数值
            f_n = self.fun(x_n, y_n)  # f_{n+1} 往前数两个节点的导数值

            x_n_2 = x_n_1 + step
            y_n_2_pre = y_n_1 + (self.step / 2) * (3 * f_n_1 - f_n)

            "==========显示2步2阶法预测n+3点函数值=========="

            #取出当前点的函数值的预测值(y_{n+2})
            f_n_2_pre = fun(x_n_2,y_n_2_pre)
            y_n_2 = y_n_1 + (self.step/2)*(f_n_2_pre+f_n_1)
            self.solve_dic[x_n_2] = y_n_2

        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Implicit 1_Step 2_Order'] = \
            {'solve': solve, 'error': error}
        return self.solve_dic

    def implicit_2k_3p(self):
        """
        显示3步3阶预估，隐式2步3阶法做校正
        :return: self.solve_dic 解集字典
        """
        # 生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[i]: self.y_samples[i] for i in range(3)}

        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            "==========显示3步3阶法预测n+3点函数值=========="

            # 计算下一个节点函数值y_{n+3}
            x_n_2 = list(self.solve_dic.keys())[-1]# x_{n+2} 往前数一个节点
            y_n_2 = list(self.solve_dic.values())[-1]#y_{n+2} 往前数一个节点的函数值
            f_n_2 = fun(x_n_2, y_n_2)  # f_{n+2} 往前数一个节点的导数值

            x_n_1 = list(self.solve_dic.keys())[-2]  # x_{n+1} 往前数两个节点
            y_n_1 = list(self.solve_dic.values())[-2]  # y_{n+1} 往前数两个节点的函数值
            f_n_1 = fun(x_n_1, y_n_1)  # f_{n+1} 往前数两个节点的导数值

            x_n = list(self.solve_dic.keys())[-3]  # x_{n} 往前数三个节点
            y_n = list(self.solve_dic.values())[-3]  # y_{n} 往前数三个节点的函数值
            f_n = fun(x_n, y_n)  # f_{n} 往前数三个节点的导数值

            x_n_3 = x_n_2 + step
            y_n_3_pre = y_n_2 +(self.step/12)*(23*f_n_2-16*f_n_1+5*f_n)

            # 取出当前点的函数值的预测值(y_{n+2})
            f_n_3_pre = fun(x_n_3, y_n_3_pre)
            y_n_3 = y_n_2 + (self.step/12)*(5*f_n_3_pre+8*f_n_2-f_n_1)
            self.solve_dic[x_n_3] = y_n_3

        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Implicit 2_Step 3_Order'] = \
            {'solve': solve, 'error': error}
        return self.solve_dic

    def implicit_3k_4p(self):
        """
        显示4步4阶预估，隐式3步4阶法做校正
        :return: self.solve_dic 解集字典
        """
        # 生成解字典，初始值是已知量
        self.solve_dic = {self.x_samples[i]: self.y_samples[i] for i in range(4)}

        # 样本点数减去已有解的数量，就是还需计算的步数
        calculate_times = len(self.x_samples) - len(self.solve_dic)

        for i in range(calculate_times):
            "==========显示4步4阶法预测n+4点函数值=========="
            # 计算下一个节点函数值y_{n+4}
            x_n_3 = list(self.solve_dic.keys())[-1]  # x_{n+3} 往前数一个节点
            y_n_3 = list(self.solve_dic.values())[-1]  # y_{n+3} 往前数一个节点的函数值
            f_n_3 = fun(x_n_3, y_n_3)  # f_{n+3} 往前数一个节点的导数值

            x_n_2 = list(self.solve_dic.keys())[-2]  # x_{n+1} 往前数两个节点
            y_n_2 = list(self.solve_dic.values())[-2]  # y_{n+1} 往前数两个节点的函数值
            f_n_2 = fun(x_n_2, y_n_2)  # f_{n+1} 往前数两个节点的导数值

            x_n_1 = list(self.solve_dic.keys())[-3]  # x_{n} 往前数三个节点
            y_n_1 = list(self.solve_dic.values())[-3]  # y_{n} 往前数三个节点的函数值
            f_n_1 = fun(x_n_1, y_n_1)  # f_{n} 往前数三个节点的导数值

            x_n = list(self.solve_dic.keys())[-4]  # x_{n} 往前数四个节点
            y_n = list(self.solve_dic.values())[-4]  # y_{n} 往前数四个节点的函数值
            f_n = fun(x_n, y_n)  # f_{n} 往前数四个节点的导数值

            x_n_4 = x_n_3 + step
            y_n_4_pre = y_n_3+(self.step/24)*(55*f_n_3-59*f_n_2+37*f_n_1-9*f_n)

            # 取出当前点的函数值的预测值(y_{n+2})
            f_n_4_pre = fun(x_n_4, y_n_4_pre)
            y_n_4 = y_n_3+(self.step /24)*(9*f_n_4_pre+19*f_n_3-5*f_n_2+f_n_1)
            self.solve_dic[x_n_4] = y_n_4

        error = self.erro_calculate()
        solve = list(self.solve_dic.values())
        self.calculate_log['Implicit 3_Step 4_Order'] = \
            {'solve': solve, 'error': error}
        return self.solve_dic

    def solve_plot(self,mode='single'):

        #多种方法多张子图画图
        if self.calculate_log and mode == 'multi':
            #n种方法误差图，1张误差对比图
            fig_number = len(self.calculate_log) + 1
            fig_index = 1
            error_dic = {}
            plt.figure(figsize=(10,2.5*fig_number))
            for method,solve_info in self.calculate_log.items():
                plt.subplot(fig_number, 1 , fig_index)
                error = solve_info['error']
                title = f"{method}  error:{error}"
                error_dic[method] = error
                plt.plot(self.x_samples, self.y_samples,
                         label='Sample Point')

                plt.plot(self.x_samples, list(solve_info['solve']),
                         label='Solve Point', ls='--')
                plt.title(title)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                fig_index += 1
            plt.subplot(fig_number, 1, fig_index)
            plt.bar(error_dic.keys(), error_dic.values(), width=0.2)
            for method,error in error_dic.items():
                plt.text(method,error,'%.3g'%(error),ha='center')
            plt.xlabel('method')
            plt.ylabel('error')
            plt.tight_layout()
            plt.show()

        # 多种方法一张主图画图
        elif self.calculate_log and mode == 'single':
            error_dic = {}
            plt.figure(figsize=(10,6))
            plt.subplot(2,1,1)
            plt.plot(self.x_samples, self.y_samples,
                     label='Sample Point')
            for method, solve_info in self.calculate_log.items():
                error = solve_info['error']
                error_dic[method] = error
                plt.plot(self.x_samples, list(solve_info['solve']),
                         label=method, ls='--')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()

            plt.subplot(2, 1, 2)
            plt.bar(error_dic.keys(),error_dic.values(),width=0.2)
            for method,error in error_dic.items():
                plt.text(method,error,'%.3g'%(error),ha='center')
            plt.xlabel('method')
            plt.ylabel('error')
            plt.tight_layout()
            plt.show()

        #如果解字典为空先进行求解
        else:
            print('请先进行求解')

    def erro_calculate(self):
        """
        计算求解点与样本点误差
        :return: error 误差
        """
        solve_point = np.array(list(self.solve_dic.values()))

        sample_point = self.y_samples
        error = np.mean((solve_point-sample_point)**2)
        return error

    def euler_predict(self,k):
        """
        欧拉法预测解，直到解的数量满足多步法的要求
        :param k: 多步法的步数，当解集中的解有k个时，就可以满足多步法计算要求
        :return: 没有返回，直接修改类属性中的solve_dic
        """
        solve_number = len(self.solve_dic)  # 目前有几个解
        #n步法，若解集中不足n个解，则用欧拉法预测解，直到解集中解的数量满足要求
        while solve_number < k:
            x_old = list(self.solve_dic.keys())[-1]
            y_old = list(self.solve_dic.values())[-1]
            solve = self.euler(x_old, y_old)

            x_new = x_old + step
            self.solve_dic[x_new] = solve
            solve_number += 1



y = lambda x:-x+np.tan(x) #原函数
fun = lambda x,y:(x+y)**2      #导数

start =0
end = 1
step = 0.1

calculator1 = Adams_method(
    y=y,
    fun=fun,
    start=start,
    end=end,
    step=step
)

calculator1.explicit_1k_1p()
# calculator1.implicit_1k_2p()
calculator1.implicit_2k_3p()
# calculator1.implicit_3k_4p()
# calculator1.explicit_2k_2p()
# calculator1.explicit_3k_3p()
# calculator1.explicit_4k_4p()

calculator1.solve_plot(mode='multi')




