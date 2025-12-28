from itertools import groupby
from util import Bar, parse_csv
from enum import Enum
from dataclasses import dataclass, field
from datetime import time


class Direction(Enum):
    UP = 1
    DOWN = 2


class Action(Enum):
    BUY = 1
    SELL = 2


@dataclass
class ProfitResult:
    purchase_price: float
    sell_price: float
    percent_gain: float = field(init=False)

    def __post_init__(self):
        self.percent_gain = (
            self.sell_price - self.purchase_price
        ) / self.purchase_price


class PercentStrategy:
    def __init__(
        self,
        percent_threshold: float,
        direction: Direction,
        action: Action,
        percent_profit_objective: float,
        time_start: time | None,
        time_end: time | None,
        stop_loss: float,
    ):
        self.percent_threshold = percent_threshold
        self.time_start = time_start
        self.direction = direction
        self.action = action
        self.percent_profit_objective = percent_profit_objective
        self.time_end = time_end
        self.stop_loss = stop_loss

        self.gains = 1.0
        self.returns: list[float] = []
        self.cumulative_returns: list[float] = []

    def __get_execution_price(self, bar: Bar, threshold: float):
        if self.direction == Direction.UP and self.action == Action.SELL:
            if bar.open >= threshold:
                return bar.open
            elif bar.open <= threshold and bar.high >= threshold:
                return threshold
            else:
                return None
        elif self.direction == Direction.DOWN and self.action == Action.BUY:
            if bar.open <= threshold:
                return bar.open
            elif bar.open >= threshold and bar.low <= threshold:
                return threshold
            else:
                return None
        elif self.action == Action.BUY:  # and self.direction == Direction.UP:
            if bar.open <= threshold and bar.high >= threshold:
                return threshold
            elif bar.open >= threshold:
                # We are buying as soon as we can once the threshold is crossed, even if it's above the threshold price
                return bar.open
            else:  # threshold price has not been reached
                return None
        else:  # self.action == Action.SELL and self.direction == Direction.DOWN:
            if bar.open >= threshold and bar.low <= threshold:
                return threshold
            elif bar.open <= threshold:
                # We are selling as soon as we can once the threshold is crossed, even if it's above the threshold price
                return bar.open
            else:
                return None

    def __get_threshold_price(self, previous_close: float) -> float:
        if self.direction == Direction.UP:
            return previous_close * (1 + self.percent_threshold)
        else:  # self.direction == Direction.DOWN:
            return previous_close * (1 - self.percent_threshold)

    def get_index_threshold_exceeded(
        self, bars: list[Bar], previous_close: float
    ) -> int | None:
        i = 0
        if self.time_start is not None:
            while i < len(bars) and bars[i].dt.time() < self.time_start:
                i += 1
        threshold = self.__get_threshold_price(previous_close)
        print(f"threshold is {threshold}")
        while i < len(bars):
            bar = bars[i]
            execution_price = self.__get_execution_price(bar, threshold)
            if execution_price is not None:
                print(f"threshold exceeded at {bar}")
                return i
            i += 1

        return None

    def get_profit(
        self, bars: list[Bar], previous_close: float, idx_threshold_exceeded: int
    ) -> ProfitResult:
        print(f"initial action time is {bars[idx_threshold_exceeded].dt}")
        execution_price = self.__get_execution_price(
            bars[idx_threshold_exceeded], self.__get_threshold_price(previous_close)
        )

        j = idx_threshold_exceeded + 1

        if self.action == Action.BUY:
            # In this scenario we already bought at the execution price and want to sell at the profit objective
            price_objective = execution_price * (1 + self.percent_profit_objective)
            stop_loss_price = execution_price * (1 - self.stop_loss)
            profit_result = None
            while j < len(bars):
                if bars[j].high >= price_objective:
                    profit_result = ProfitResult(execution_price, price_objective)
                    break
                elif bars[j].low <= stop_loss_price:
                    profit_result = ProfitResult(execution_price, stop_loss_price)
                    break
                j += 1
            if profit_result is None:
                profit_result = ProfitResult(execution_price, bars[-1].open)
        else:  # self.action == Action.SELL:
            # In this scenario we already sold at the execution price and want to buy at the profit objective
            price_objective = execution_price * (1 - self.percent_profit_objective)
            stop_loss_price = execution_price * (1 + self.stop_loss)
            profit_result = None
            while j < len(bars):
                if bars[j].low <= price_objective:
                    profit_result = ProfitResult(price_objective, execution_price)
                    break
                elif bars[j].high >= stop_loss_price:
                    profit_result = ProfitResult(stop_loss_price, execution_price)
                    break
                j += 1
            if profit_result is None:
                profit_result = ProfitResult(bars[-1].open, execution_price)
        print(profit_result)
        print(f"final action time is {bars[j if j < len(bars) else -1].dt}")
        return profit_result

    def analyze_bars(self, bars: list[Bar], previous_close: float) -> None:
        idx_threshold_exceeded = self.get_index_threshold_exceeded(bars, previous_close)
        if idx_threshold_exceeded is None:
            return
        profit_result = self.get_profit(bars, previous_close, idx_threshold_exceeded)
        self.returns.append(profit_result.percent_gain)
        self.gains *= 1.0 + profit_result.percent_gain
        self.cumulative_returns.append(self.gains)

    def run_strategy_by_day(self, bars: list[Bar]) -> None:
        bars.sort(key=lambda x: x.dt)
        grouped = groupby(bars, lambda x: x.dt.date())
        first_day = next(grouped)
        *_, last = first_day[1]
        previous_close = last.close
        for count, (date, group) in enumerate(grouped):
            print(f"day {count + 1}")
            print(f"previous close: {previous_close}")
            bars = list(group)
            self.analyze_bars(bars, previous_close)
            previous_close = bars[-1].close

        print(f"Final gains: {self.gains}")


if __name__ == "__main__":
    # file_path = "historical_data_WBA_5_mins.csv"
    file_path = "alpaca_historical_data_SPY_20150101_20250505.csv"
    bars = parse_csv(file_path)
    bars = [
        bar
        for bar in bars
        if bar.dt.time() >= time(9, 30) and bar.dt.time() < time(15, 59)
    ]
    minute_percent_change = [(bar.high - bar.low) / bar.open for bar in bars]
    import signal
    import matplotlib.pyplot as plt

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.plot(minute_percent_change)
    plt.show()
    # percent_strategy = PercentStrategy(percent_threshold=0.004, direction=Direction.DOWN, action=Action.BUY, percent_profit_objective=0.10, time_start=None, time_end=None, stop_loss=0.002)
    # percent_strategy.run_strategy_by_day(bars)
    # print(percent_strategy.gains)
    # import matplotlib.pyplot as plt
    # import signal
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    # plt.plot(percent_strategy.returns)
    # plt.plot(percent_strategy.cumulative_returns)
    # plt.show()
