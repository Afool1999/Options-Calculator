import math
import numpy as np
from ._util import Mix_Norm_Rnd, Double_Expo_Rnd
from numpy.random import normal, poisson

class JumpParams:
    def __init__(self) -> None:
        self.kappa = 0.
        self.lam = 0.

class  MonteCarloJumpDiffusion:
    def __init__(self, option="asian", sigma=0.2, jump_model="no", n_sim=10000, mult=2) -> None:
        self.price_func_dict = {
            "eu": None,
            "american": None,
            "asian": self.__Price_MC_Asian_Strikes_func,
            "barrier": self.__Price_MC_Barrier_Strikes_func,
        }
        assert(option in self.price_func_dict)
        self.option = option
        self.price_func = self.price_func_dict[option]
        self.sigma = sigma
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

    def price(self, S_0, r, q, T, M, Kvec=None, type="call", down=1, H=.85, rebate=0.):
        if Kvec is None:
            Kvec = S_0 * np.array([.85, .90, .95, 1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.5, 1.6])
        
        M_mult = M * self.mult
        simulate_path = self.__Simulate_Jump_Diffusion_func(n_sim=self.n_sim, 
                                                     M=M_mult + 1, 
                                                     T=T, 
                                                     S_0=S_0, 
                                                     r=r, 
                                                     q=q, 
                                                     sigma=self.sigma, 
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
        
