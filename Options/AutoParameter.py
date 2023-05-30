from yahoo_fin import stock_info as si
import yfinance as yf
import datetime
from datetime import date
import math
import numpy as np
import akshare as ak
from scipy.stats import norm
from math import log, sqrt, pi, exp


def d(P, X, T, r, s):
    d1 = (log(P / X) + (r + s ** 2 / 2) * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    return d1, d2
    # returns Black-Scholes call price


def call(P, X, T, r, s):
    d1, d2 = d(P, X, T, r, s)
    C = P * norm.cdf(d1) - X * exp(-r * T) * norm.cdf(d2)
    return C
    # uses iterative guessing i.e. Newton's method to estimate implied volatility


def call_iv(P, X, T, r, C):
    # guesses for volatility - we run Newton's method on multiple guesses to increase likelihood of convergence
    s_list = [x * 0.005 for x in range(10, 300, 5)]

    for s in s_list:
        # defining error and bounds - one for convergence and divergence
        err = 1
        good_bound = 0.000001
        bad_bound = 100
        success = True
        while (err >= good_bound):
            if (err >= bad_bound or s <= 0):
                success = False
                break

            diff = call(P, X, T, r, s) - C
            d1, d2 = d(P, X, T, r, s)
            vega = P * norm.pdf(d1) * sqrt(T)
            if vega < 0.001:
                success = False
                break
            s = s - (diff / vega)
            err = abs(diff)

        if success:
            return s
    return 0

class ParameterGen:
    def __init__(self):
        self.stock=None
        self.start_date = datetime.date.today() - datetime.timedelta(days=365)
        self.r = self.__get_rate()

    def __get_rate(self):
        try:
            r = ak.bond_zh_us_rate(start_date=str(self.start_date).replace('-', ''))['美国国债收益率2年']
            r = np.nanmean(r) / 100.
        except:
            r = 0.05
        return r

    def __get_q(self):
        try:
            q = float(si.get_dividends(self.stock).tail(1).iloc[0]["dividend"]) / 100
        except:
            q = 0.03
        return q

    def __get_implied_volatility(self):
        try:
            ticker = yf.Ticker(self.stock)
            if not ticker.options:
                return 0.2
            expirys = []
            moneynesses = []
            ivs = []
            today = date.today()
            for d in ticker.options:
                # Find time to expiry date
                year, month, day = d.split("-")
                date_parsed = date(int(year), int(month), int(day))
                T = (date_parsed - today).days / 365.0
                if T == 0:
                    continue
                calls = ticker.option_chain(d)[0]
                price = float(ticker.history(d).drop_duplicates(subset=['Close'])['Close'])
                for _, call in calls.iterrows():
                    iv = call_iv(price, call['strike'], T, self.r, call['lastPrice'])
                    moneyness = call['strike'] / price
                    # calc_iv == 0 implies that the call_iv failed to converge to a positive volatility
                    # checks for reasonable bounds on values
                    if iv != 0 and 0 < moneyness and moneyness < 2:
                        expirys.append(T)
                        moneynesses.append(moneyness)
                        ivs.append(iv)
            return sum(ivs)/len(ivs)/4
        except:
            return 0.2
    def get_para(self,stock:str):
        self.stock=stock
        para_dict={
            "n_sim": 10000,
            "r": self.__get_rate(),
            "q": self.__get_q(),
            "jump_model":"no",
            "sigma":self.__get_implied_volatility()
        }
        return  para_dict

if __name__=='__main__':
    G=ParameterGen()
    print(G.get_para('nflx'))
