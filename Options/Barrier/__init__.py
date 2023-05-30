from ..MonteCarlo._MonteCarlo import MonteCarloJumpDiffusion

class MonteCarloBarrierJump(MonteCarloJumpDiffusion):
    def __init__(self, jump_model="no", n_sim=10000, mult=2) -> None:
        MonteCarloJumpDiffusion.__init__(self, option="barrier", jump_model=jump_model, n_sim=n_sim, mult=mult)
        

__all__ = [
    "MonteCarloBarrierJump",
]