# Writer:Hdu_WangXu
# Time: 2020-06-05
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


class Learner(object):
    def __init__(self, predict_range, e_0, i_0, r_0, N, ratio):
        self.predict_range = predict_range
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.N = N
        self.s_0 = N - e_0 - i_0 - r_0
        self.ratio = ratio

    def __load_data__(self, root):
        df = pd.read_csv(root)
        df.set_index(["Date"], inplace=True)
        confirmed = df["Confirmed"].T
        exposed = confirmed * self.ratio
        recovered = df["Recovered"].T
        dead = df["Dead"].T
        susceptible = self.N - confirmed - exposed - recovered - dead
        return susceptible, exposed, confirmed, recovered, dead


    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, alpha, gamma, data, exposed, recovered, death, s_0, e_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        N = self.N

        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta * S * I / N, beta * S * I / N - alpha * E, alpha * E - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_exposed = np.concatenate((exposed.values, [None] * (size - len(exposed.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_exposed, extended_recovered, extended_death, solve_ivp(SEIR,[0, size],
                                                                                         [s_0, e_0,i_0, r_0],
                                                                                         t_eval=np.arange(0, size, 1))

    def train(self):
        susceptible, exposed, confirmed, recovered, dead = self.__load_data__('./Data/US_All.csv')

        optimal = minimize(self.loss, np.array([0.001, 0.001, 0.001]),
                           args=(susceptible, exposed, recovered, self.s_0, self.e_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.6), (0.00000001, 0.4), (0.00000001, 0.4)])

        print(optimal)

        '''预测数据'''
        beta, alpha, gamma = optimal.x
        new_index, extended_actual, extended_exposed, extended_recovered, extended_death, prediction = \
            self.predict(beta, alpha, gamma, susceptible, exposed, recovered, dead,
                                             self.s_0, self.e_0, self.i_0, self.r_0)
        df = pd.DataFrame(
            {
            # 'Infected data': extended_actual,
             'Recovered data': extended_recovered,
             'Death data': extended_death,
             #'Susceptible': prediction.y[0],
             'Exposed': prediction.y[1],
             'Infected': prediction.y[2],
             'Recovered': prediction.y[3]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        #ax.set_title(self.country)
        df.plot(ax=ax)
        print(f" beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.savefig("result_SEIR.png")
        fig.show()
        print(df)
        df.to_csv('result_SEIR.csv')

    def loss(self, point, susceptible, exposed, recovered, s_0, e_0, i_0, r_0):
        size = len(susceptible)
        beta, alpha, gamma = point
        N = self.N


        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta * S * I / N, beta * S * I / N - alpha * E, alpha * E - gamma * I, gamma * I]

        solution = solve_ivp(SEIR, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)
        l1 = np.sqrt(np.mean((solution.y[1] - susceptible) ** 2))
        l2 = np.sqrt(np.mean((solution.y[2] - exposed) ** 2))
        l3 = np.sqrt(np.mean((solution.y[3] - recovered) ** 2))
        a1 = 0.1
        a2 = 0.1
        return a1 * l1 + a2 * l2 + (1 - a1 - a2) * l3




predict_range = 30
N = 3.282 * 10 ** 9
s_0 = 99990  # 易感人数
e_0 = 8  # 潜伏者
i_0 = 2  # 感染者
r_0 = 0  # 康复者
ratio = 0.3

learner = Learner(predict_range, e_0, i_0, r_0, N, ratio)
learner.train()
