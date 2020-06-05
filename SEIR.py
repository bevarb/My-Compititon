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

    def __load_data__(self, root, begain, over):
        df = pd.read_csv(root)
        df.set_index(["Date"], inplace=True)
        self.Real_data = df
        confirmed = df.loc[begain:over, "Confirmed"].T  # 所有确诊的，后面尽管变为R or D ，数目是累计的
        exposed = df.loc[begain:over, "Exposed"].T  # 将确诊的往前推5天作为Exposed
        recovered = df.loc[begain:over, "Recovered"].T
        dead = df.loc[begain:over, "Dead"].T
        susceptible = np.ones(len(confirmed)) * self.N - confirmed - exposed - recovered - dead
        return susceptible, exposed, confirmed, recovered, dead


    def extend_index(self, index, new_size):
        '''
        extend index for plot
        :param index: smaller index
        :param new_size: bigger index
        :return:
        '''
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def extend_data(self, data):
        new_index = self.extend_index(data.index, len(data.index) + self.predict_range)
        size = len(new_index)
        extended = np.concatenate((data.values, [None] * (size - len(data.values))))
        return extended

    def predict(self, beta, alpha, gamma, size, s_0, e_0, i_0, r_0):
        '''input parameters to predict'''

        N = self.N
        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta * S * I / N, beta * S * I / N - alpha * E, alpha * E - gamma * I, gamma * I]

        return solve_ivp(SEIR, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1))

    def train(self, root, start, over):
        susceptible, exposed, confirmed, recovered, dead = self.__load_data__(root, start, over)

        optimal = minimize(self.loss, np.array([0.2, 0.2, 0.2]),
                           args=(susceptible, exposed, confirmed - recovered - dead, recovered + dead,
                                 self.s_0, self.e_0, self.i_0, self.r_0),
                           method='L-BFGS-B', bounds=[(0.00000001, 0.8), (0.14, 0.33), (0.35, 0.6)])

        print(optimal)

        new_index = self.extend_index(susceptible.index, len(susceptible.index) + self.predict_range)
        '''Predict'''
        beta, alpha, gamma = optimal.x
        print(f" beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        '''Plot'''
        prediction = self.predict(beta, alpha, gamma, len(new_index), self.s_0, self.e_0, self.i_0, self.r_0)
        # extend all data
        extended_confirmed = self.extend_data(confirmed)
        extended_recovered = self.extend_data(recovered)
        extended_death = self.extend_data(dead)
        # extended_confirmed = self.extend_data(self.Real_data["Confirmed"].T)
        # extended_recovered = self.extend_data(self.Real_data["Recovered"].T)
        # extended_death = self.extend_data(self.Real_data["Dead"].T)
        df = pd.DataFrame(
            {
             'Real Infected': extended_confirmed,
             'Real Recovered': extended_recovered,
             'Real Death': extended_death,
             #'Susceptible': prediction.y[0],
             #'Predict Exposed': prediction.y[1],
             'Predict Infected': prediction.y[2],
             'Rredict Recovered': prediction.y[3]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        #ax.set_title(self.country)
        df.plot(ax=ax)
        fig.savefig("result_SEIR.png")
        fig.show()
        df.to_csv('result_SEIR.csv')

    def loss(self, point, susceptible, exposed, infected, removed, s_0, e_0, i_0, r_0):
        size = len(susceptible)
        beta, alpha, gamma = point
        N = self.N

        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta * S * (I + E) / N, beta * S * I / N - alpha * E, alpha * E - gamma * I, gamma * I]

        solution = solve_ivp(SEIR, [0, size], [s_0, e_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)
        # l1 = np.sqrt(np.mean((solution.y[0] - susceptible) ** 2))
        l2 = np.sqrt(np.mean((solution.y[1] - exposed) ** 2))
        l3 = np.sqrt(np.mean((solution.y[2] - infected) ** 2))
        l4 = np.sqrt(np.mean((solution.y[3] - removed) ** 2))
        a1 = 0.4
        a2 = 0.4
        a3 = 0.1
        return a2 * l2 + a3 * l3 + (1 - a1 - a2 - a3) * l4




predict_range = 30
N = 3.282 * (10 ** 9)

e_0 = 2  # 潜伏者
i_0 = 4  # 感染者
r_0 = 0  # 康复者
ratio = 0.25

learner = Learner(predict_range, e_0, i_0, r_0, N, ratio)
learner.train('./Data/Us_All.csv', "3/22/20", "5/29/20")
