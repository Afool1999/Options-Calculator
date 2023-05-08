from yahoo_fin import stock_info as si
from yahoo_fin import options
import datetime
import math
import numpy as np
import akshare as ak
from scipy.stats import norm
from math import log, sqrt, pi, exp
from pandas import DataFrame
import matplotlib.pyplot as plt
 
class BlackSholes:
    def __init__(self) -> None:
        pass

    def d1(self, S, K, T, r, sigma):
        return(log(S / K) + (r + sigma**2 / 2.) * T) / sigma * sqrt(T)
    
    def d2(self, S, K, T, r, sigma):
        return self.d1(S, K, T, r, sigma) - sigma * sqrt(T)

    ## define the call options price function
    def bs_call(self, S, K, T, r, sigma):
        return S * norm.cdf(self.d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(self.d2(S, K, T, r, sigma))
    
    ## define the put options price function
    def bs_put(self, S, K, T, r, sigma):
        return K * exp(-r * T) - S + self.bs_call(S, K, T, r, sigma)
    
    ## define the Call_Greeks of an option
    def call_delta(self, S, K, T, r, sigma):
        return norm.cdf(self.d1(S, K, T, r, sigma))
    
    def call_gamma(self, S, K, T, r, sigma):
        return norm.pdf(self.d1(S, K, T, r, sigma))/(S * sigma * sqrt(T))
    
    def call_vega(self, S, K, T, r, sigma):
        return 0.01 * (S * norm.pdf(self.d1(S, K, T, r, sigma)) * sqrt(T))
    
    def call_theta(self, S, K, T, r, sigma):
        return 0.01 * (-(S * norm.pdf(self.d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T)) - r * K
         * exp(-r*T)*norm.cdf(self.d2(S, K, T, r, sigma)))
    
    def call_rho(self, S, K, T, r, sigma):
        return 0.01 * (K * T * exp(-r * T) * norm.cdf(self.d2(S, K, T, r, sigma)))
    
    ## define the Put_Greeks of an option
    def put_delta(self, S, K, T, r, sigma):
        return -norm.cdf(-self.d1(S, K, T, r, sigma))
    def put_gamma(self, S, K, T, r, sigma):
        return norm.pdf(self.d1(S, K, T, r, sigma))/(S * sigma * sqrt(T))
    def put_vega(self, S, K, T, r, sigma):
        return 0.01*(S*norm.pdf(self.d1(S, K, T, r, sigma)) * sqrt(T))
    def put_theta(self, S, K, T, r, sigma):
        return 0.01 * (-(S * norm.pdf(self.d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-self.d2(S, K, T, r, sigma)))
    def put_rho(self, S, K, T, r, sigma):
        return 0.01*(-K*T*exp(-r*T)*norm.cdf(-self.d2(S, K, T, r, sigma)))

    def statistics(self, S, K, T, r, sigma):
        price_and_greeks = {'Call' : [self.bs_call(S, K, T, r, sigma), self.call_delta(S, K, T, r, sigma), self.call_gamma(S, K, T, r, sigma),self.call_vega(S, K, T, r, sigma), self.call_rho(S, K, T, r, sigma), self.call_theta(S, K, T, r, sigma)],'Put' : [self.bs_put(S, K, T, r, sigma), BS.put_delta(S, K, T, r, sigma), self.put_gamma(S, K, T, r, sigma), self.put_vega(S, K, T, r, sigma), self.put_rho(S, K, T, r, sigma), self.put_theta(S, K, T, r, sigma)]}
        price_and_greeks_frame = DataFrame(price_and_greeks, columns=['Call','Put'], index=['Price', 'delta', 'gamma','vega','rho','theta'])
        return price_and_greeks_frame


# tickers_nasdap = si.tickers_nasdaq()
# tickers_dow = si.tickers_dow()

# print(tickers_nasdap)

def split_line(words):
    print("-" * 10 + words + "-" * 10)


ticker = input("Input ticker (e.g. nflx, aapl): ")
# option_type = input("Input option type (calls/puts): ")

split_line("Stock Info (recent 3 days)")
data = si.get_data(ticker)
print(data[-3:])

dates = options.get_expiration_dates(ticker)
split_line("Expiration Dates")
print("\n".join(dates))

date = input("Input expiration date (from above): ")
ticker_options = options.get_options_chain(ticker, date)
split_line("Contracts of Expiration Date {}".format(date))
split_line("Calls")
print(ticker_options["calls"])
split_line("Puts")
print(ticker_options["puts"])
K = float(input("Input strike price: "))


start_date = datetime.date.today() - datetime.timedelta(days=365)
hist_data = si.get_data(ticker, start_date=start_date)
def get_sigma(data):
    X = []
    for i in range(len(data)-1):
        X.append(math.log(data[i+1]) - math.log(data[i]))
    s = np.std(X, ddof=1)
    return s * math.sqrt(252)

sigma = get_sigma(hist_data['open']) 
S = si.get_live_price(ticker)
r = ak.bond_zh_us_rate(start_date=str(start_date).replace('-', ''))['美国国债收益率2年']
r = np.nanmean(r) / 100.
expiration_date = datetime.datetime.strptime(date, "%B %d, %Y")
T = (expiration_date - datetime.datetime.utcnow()).days / 365.
BS = BlackSholes()

sta = BS.statistics(S, K, T, r, sigma)
split_line("Statistics of BS Model")
print(sta)



# manually set an expiration date
# fix T and draw figure

date = "May 5, 2023"
expiration_date = datetime.datetime.strptime(date, "%B %d, %Y")
ticker_options = options.get_options_chain(ticker, date)["calls"]
T = (expiration_date - datetime.datetime.utcnow()).days / 365.

x = []
y_call = []
y_put = []
for option in np.array(ticker_options).tolist():
    K = option[2]
    x.append(K)
    call_price = BS.bs_call(S, K, T, r, sigma)
    put_price = BS.bs_put(S, K, T, r, sigma)
    y_call.append(call_price)
    y_put.append(put_price)

plt.plot(x, y_call, label='Call', color='r')
plt.plot(x, y_put, label='Put', color='orange')
plt.vlines(S, min(y_call[-1], y_put[0]), max(y_put[-1], y_call[0]), linestyles='-.', colors='g', label='Current Stock Price')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.legend()
plt.savefig('./fig/fig1.png')
plt.cla()


# fix K
K = S
x = []
y_call = []
y_put = []
for date in dates:
    expiration_date = datetime.datetime.strptime(date, "%B %d, %Y")
    x.append(expiration_date.strftime('%y-%m-%d'))
    # ticker_options = options.get_options_chain(ticker, date)[option_type]
    T = (expiration_date - datetime.datetime.utcnow()).days / 365.
    call_price = BS.bs_call(S, K, T, r, sigma)
    put_price = BS.bs_put(S, K, T, r, sigma)
    y_call.append(call_price)
    y_put.append(put_price)

plt.plot(x, y_call, label='Call', color='r')
plt.plot(x, y_put, label='Put', color='orange')
# plt.vlines(S, min(y_call[-1], y_put[0]), max(y_put[-1], y_call[0]), linestyles='-.', colors='g', label='Current Stock Price')
plt.xlabel('Date')
plt.ylabel('Option Price')
plt.xticks(range(0, len(x), 3))
plt.legend()
plt.savefig('./fig/fig2.png')
plt.cla()