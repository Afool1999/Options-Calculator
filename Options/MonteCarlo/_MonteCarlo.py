import math
import numpy as np
from ._util import Mix_Norm_Rnd, Double_Expo_Rnd
from numpy.random import normal, poisson

class JumpParams:
    def __init__(self) -> None:
        self.kappa = 0.
        self.lam = 0.

class  MonteCarloJumpDiffusion:
    def __init__(self, option="asian", jump_model="no", n_sim=10000, mult=2) -> None:
        """
        初始化蒙特卡洛模拟的参数。

        输入参数
        ----------
        option : str, default = "asian"
            期权种类，option in ["asian", "eu", "american", "barrier"]。
        
        jump_model : str, default = "no"
            选取跳跃模型，默认不使用，jump_model in ["no", "normal", "double", "mix"]
        
        n_sim : int, default = 10000
            蒙特卡洛模拟次数。
        
        mult : int, default = 2
            模拟精细程度，实际模拟步数为 M * mult。
        """
        self.price_func_dict = {
            "eu": None,
            "american": None,
            "asian": self.__Price_MC_Asian_Strikes_func,
            "barrier": self.__Price_MC_Barrier_Strikes_func,
        }
        assert(option in self.price_func_dict)
        self.option = option
        self.price_func = self.price_func_dict[option]
        self.jump_models = [
            "no",
            "normal",
            "double",
            "mix",
        ]
        assert(jump_model in self.jump_models)
        self.jump_model = jump_model
        self.jump_params = JumpParams()
        if jump_model == "normal":
            lam     = 1.
            muJ     = -.10
            sigJ    = 0.3

            self.jump_params.kappa = math.exp(muJ + .5 * (sigJ ** 2)) - 1
            self.jump_params.lam = lam
            self.jump_params.muJ = muJ
            self.jump_params.sigJ = sigJ
        elif jump_model == "double":
            lam     = 1.
            p_up    = .5 # up jump probability    
            eta1    = 25.
            eta2    = 30.
            
            kappa  = p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1
            self.jump_params.lam = lam
            self.jump_params.kappa = kappa
            self.jump_params.eta1 = eta1
            self.jump_params.eta2 = eta2
            self.jump_params.p_up = p_up
        elif jump_model == "mix":
            lam = 1. 
            a1 = -0.05
            b1 = 0.07    
            a2 = 0.02 
            b2 = 0.03
            p_up = 0.6

            kappa = p_up * math.exp(a1 + .5 * (b1**2)) + (1 - p_up) * math.exp(a2 + .5 * (b2**2)) - 1
            self.jump_params.lam = lam
            self.jump_params.kappa = kappa
            self.jump_params.a1 = a1
            self.jump_params.b1 = b1
            self.jump_params.a2 = a2
            self.jump_params.b2 = b2
            self.jump_params.p_up = p_up

        self.n_sim = n_sim
        self.mult = mult
    
    def __Simulate_Jump_Diffusion_func(self, n_sim, M, T, S_0, r, q, sigma, jump_model, jump_params):
        dt = T / M
        
        if jump_model != "no":
            lam = jump_params.lam
            kappa = jump_params.kappa

            zeta = r - q - lam * kappa
            lamt = lam * dt

            if jump_model == "normal":
                muJ = jump_params.muJ
                sigJ = jump_params.sigJ
                jump_func = lambda n: np.sum(muJ + sigJ * normal(loc=0., scale=1., size=n))
            elif jump_model == "double":
                p_up = jump_params.p_up
                eta1 = jump_params.eta1
                eta2 = jump_params.eta2
                jump_func = lambda n: np.sum(Double_Expo_Rnd(n, p_up, eta1, eta2))
            elif jump_model == "mix":
                p_up = jump_params.p_up
                a1 = jump_params.a1
                a2 = jump_params.a2
                b1 = jump_params.b1
                b2 = jump_params.b2
                jump_func = lambda n: np.sum(Mix_Norm_Rnd(n, p_up, a1, b1, a2, b2))
        else:
            zeta = r - q
        
        
        sim_path = np.zeros([n_sim, M + 1])
        sim_path[:, 0] = S_0
        sigma_sqrtdt = sigma * math.sqrt(dt)
        drift = (zeta - 0.5 * (sigma ** 2)) * dt

        if jump_model == "no":
            for step in range(1, M + 1):
                W1 = normal(loc=0, scale=1, size=n_sim)
                sim_path[:, step] = sim_path[:, step - 1] * np.exp(drift + sigma_sqrtdt * W1)

        else:
            for step in range(1, M + 1):
                poi = poisson(lam=lamt, size=n_sim)
                jump = np.zeros(n_sim)
                for sim_id in range(n_sim):
                    if poi[sim_id] > 0:
                        jump[sim_id] = jump_func(poi[sim_id])

                W1 = normal(loc=0, scale=1, size=n_sim)
                sim_path[:, step] = sim_path[:, step - 1] * np.exp(drift + jump + sigma_sqrtdt * W1)
        
        return sim_path

    def __Price_MC_Asian_Strikes_func(self, simulate_path, type, Kvec, M, mult, r, T):
        prices = np.zeros(len(Kvec))
        std_errs = np.zeros(len(Kvec))
        n_sim = len(simulate_path)
        disc = math.exp(-r * T)

        avg = np.mean(simulate_path[:, 1::mult], axis=1)
        for i in range(len(Kvec)):
            K = Kvec[i]
            if type == "call":
                payoffs = np.maximum(avg - K, 0)
            else:
                payoffs = np.maximum(K - avg, 0)
            
            prices[i] = disc * np.mean(payoffs)
            std_errs[i] = disc * np.std(payoffs) / math.sqrt(n_sim)
        
        return prices, std_errs
    
    def __Price_MC_EU_Strikes_func(self, simulate_path, type, Kvec, M, mult, r, T):
        prices = np.zeros(len(Kvec))
        std_errs = np.zeros(len(Kvec))
        n_sim = len(simulate_path)
        disc = math.exp(-r * T)

        St = simulate_path[:, mult]
        for i in range(len(Kvec)):
            K = Kvec[i]
            if type == "call":
                payoffs = np.maximum(St - K, 0)
            else:
                payoffs = np.maximum(K - St, 0)
            
            prices[i] = disc * np.mean(payoffs)
            std_errs[i] = disc * np.std(payoffs) / math.sqrt(n_sim)
        
        return prices, std_errs
    
    def __Price_MC_Barrier_Strikes_func(self, simulate_path, type, Kvec, M, mult, r, T, down, H, rebate):
        prices = np.zeros(len(Kvec))
        std_errs = np.zeros(len(Kvec))
        n_sim = len(simulate_path)
        disc = math.exp(-r * T)
        knock_time = np.zeros(n_sim)

        dt = T / M
        H = simulate_path[0, 0] * H

        if down == 1:
            for sim_id in range(n_sim):
                indices = np.where(simulate_path[sim_id, 1::mult] < H)[0]
                if len(indices) > 0:
                    knock_time[sim_id] = indices[0] * dt
                else:
                    knock_time[sim_id] = 0.
        else:
            for sim_id in range(n_sim):
                indices = np.where(simulate_path[sim_id, 1::mult] > H)[0]
                if len(indices) > 0:
                    knock_time[sim_id] = indices[0] * dt
                else:
                    knock_time[sim_id] = 0.
        
        if rebate > 0:
            disc_rebate = rebate * np.exp(-r * knock_time) * (knock_time > 0)
        else:
            disc_rebate = 0.


        for i in range(len(Kvec)):
            K = Kvec[i]
            if type == "call":
                payoffs = np.maximum(simulate_path[:, -1] - K, 0) * (knock_time == 0) + disc_rebate
            else:
                payoffs = np.maximum(K - simulate_path[:, -1], 0) * (knock_time == 0) + disc_rebate
            
            prices[i] = disc * np.mean(payoffs)
            std_errs[i] = disc * np.std(payoffs) / math.sqrt(n_sim)
        
        return prices, std_errs

    
    def price(self, S_0: float, r: float, q: float, T: float=1, M: int=252, sigma=0.2, Kvec=None, type:str="call", down:int=1, H:float=.85, rebate:float=0., polyOrder:int=3):
        """蒙特卡洛模拟，计算期权定价的函数。

        输入参数
        ----------
        S_0 : float
            标的物当前价格（例如股价、黄金现价等）。

        r : float
            无风险利率（年）。

        q : float
            股息率。

        T : float, default = 1
            期限（年），默认 T = 1，即一年期期权。

        M : int, default = 252
            交易日数，默认一年 252 天。
        
        sigma : float, default = 0.2
            股票价格波动率

        Kvec : list, default = None
            默认 Kvec = [.85, .90, .95, 1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.5, 1.6], 分别计算 S_0 * Kvec[i] 为行权价时的期权定价。
        
        type : str, default = "call"
            计算看涨或看跌期权, type in ["call", "put"]。
        
        down : int, default = 1
            barrier 期权的参数，down = 1 表式 down-and-out option，否则为 up-and-out option。
        
        H : float, default = 0.85
            barrier 期权的参数，down-and-out 或者 up-and-out 的界。
        
        rebate : float, default = 0.
            barrier 期权的参数，达到 H * S_0 时的退款。
        
        polyOrder : int, default = 3.
            美式期权参数，order of polynomial regression of continuation val。

        返回值
        -------
        prices : ndarray of shape (len(Kvec), )
            每个期权的定价。
        
        std_errs : ndarray of shape (len(Kvec), )
            期权定价的标准差。
        
        simulate_path : ndarray of shape (n_sim, M + 1)
            n 条模拟标的物价格的轨迹。
        """
        if Kvec is None:
            Kvec = S_0 * np.array([.85, .90, .95, 1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.5, 1.6])
        
        M_mult = M * self.mult
        simulate_path = self.__Simulate_Jump_Diffusion_func(n_sim=self.n_sim, 
                                                     M=M_mult + 1, 
                                                     T=T, 
                                                     S_0=S_0, 
                                                     r=r, 
                                                     q=q, 
                                                     sigma=sigma, 
                                                     jump_model=self.jump_model, 
                                                     jump_params=self.jump_params)
        if self.option == "asian":
            prices, std_errs = self.__Price_MC_Asian_Strikes_func(simulate_path=simulate_path, 
                                                                  type=type, 
                                                                  Kvec=Kvec, 
                                                                  M=M, 
                                                                  mult=self.mult,
                                                                  r=r,
                                                                  T=T)
        elif self.option == "eu":
            prices, std_errs = self.__Price_MC_EU_Strikes_func(simulate_path=simulate_path, 
                                                                  type=type, 
                                                                  Kvec=Kvec, 
                                                                  M=M, 
                                                                  mult=self.mult,
                                                                  r=r,
                                                                  T=T)
        elif self.option == "american":
            prices, std_errs = self.__Price_MC_American_Strikes_func(simulate_path=simulate_path, 
                                                                  type=type, 
                                                                  Kvec=Kvec, 
                                                                  M=M, 
                                                                  mult=self.mult,
                                                                  r=r,
                                                                  T=T,
                                                                  polyOrder=polyOrder)
        elif self.option == "barrier":
            prices, std_errs = self.__Price_MC_Barrier_Strikes_func(simulate_path=simulate_path, 
                                                                  type=type, 
                                                                  Kvec=Kvec, 
                                                                  M=M, 
                                                                  mult=self.mult,
                                                                  r=r,
                                                                  T=T,
                                                                  down=down,
                                                                  H=H,
                                                                  rebate=rebate)
        return prices, std_errs, simulate_path
        
