from stable_baselines3.common.callbacks import BaseCallback


class LapCounterCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.total_laps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            laps = info.get("lap_count", 0)
            if laps > 0:
                self.total_laps += laps

        if self.n_calls % 1000 == 0:
            self.logger.record("custom/lap_count", self.total_laps)
        return True
