from Options import MonteCarloJumpDiffusion
from Options.Asian import MonteCarloAsianJump
from Options.Barrier import MonteCarloBarrierJump
import matplotlib.pyplot as plt

model = MonteCarloJumpDiffusion(n_sim=1000, option="american", jump_model="mix")
prices, std_errs, sim_path = model.price(100, 0.05, 0.03, 1, 252, down=1, H=0.85, rebate=5.)
print(prices)

# print(sim_path[:2])
plt.plot(sim_path.transpose(),lw=1.5)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
