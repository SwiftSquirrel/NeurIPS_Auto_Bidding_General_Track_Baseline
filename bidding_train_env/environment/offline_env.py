import numpy as np


class OfflineEnv:
    """
    Simulate an advertising bidding environment.
    """

    def __init__(self, min_remaining_budget: float = 0.1):
        """
        Initialize the simulation environment.
        :param min_remaining_budget: The minimum remaining budget allowed for bidding advertiser.
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pValues: np.ndarray,pValueSigmas: np.ndarray, bids: np.ndarray, leastWinningCosts: np.ndarray, costs: np.ndarray):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        # cost for each slot
        costs = costs.reshape(-1, 3)
        # winned slot
        slot_status = np.where(bids >= costs[:, 0], 1,
                        np.where(bids >= costs[:, 1], 2,
                                np.where(bids >= costs[:, 2], 3, 0)))

        # the cost after exposure
        slot_cost = np.where(slot_status == 1, costs[:, :, 0], 0)
        slot_cost += np.where(slot_status == 2, costs[:, :, 1], 0)
        slot_cost += np.where(slot_status == 3, costs[:, :, 2], 0)
        # tick_p: exposure prob, based on historic stats
        tick_p = np.where(slot_status==1, 1,
                          np.where(slot_status==2, 0.79957,
                                   np.where(slot_status==3, 0.48233, 0)))
        # tick_status: exposure status
        tick_status = np.random.binomial(n=1, p=tick_p)
        # cost for each p
        tick_cost = slot_cost * tick_status

        # conversion rate
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values*tick_status
        tick_value = np.clip(values,0,1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status, tick_conversion, slot_status




def test():
    pv_values = np.array([10, 20, 30, 40, 50])
    pv_values_sigma = np.array([1, 2, 3, 4, 5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status,tick_conversion = env.simulate_ad_bidding(pv_values, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
