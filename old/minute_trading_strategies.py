from util import parse_csv, Bar
from enum import Enum
from itertools import groupby
from datetime import time
import matplotlib.pyplot as plt
import signal
from dataclasses import dataclass, field
from functools import reduce
import operator

signal.signal(signal.SIGINT, signal.SIG_DFL)


def calculate_odds(bars: list[Bar], threshold: float, window: int):
    if len(bars) < 2:
        return 0.0
    i = 0
    count = 0
    while i < len(bars) - window + 1:
        current = bars[i]
        current_close = current.close
        window_high = max(bar.high for bar in bars[i + 1 : i + window + 1])
        if (window_high - current_close) / current_close >= threshold:
            count += 1
        i += 1
    return count / (len(bars) - window)


@dataclass
class StrategyResult:
    bars: list[Bar]
    gains: list[float]
    compounded_results: float = field(init=False)
    sum_gains: float = field(init=False)

    def __post_init__(self):
        self.compounded_results = reduce(
            operator.mul, [1 + gain for gain in self.gains]
        )
        self.sum_gains = reduce(operator.add, self.gains)


def is_green(bar: Bar):
    return bar.open > bar.close


def is_red(bar: Bar):
    return bar.open < bar.close


def simulate_long_strategy(bars: list[Bar], threshold: float, window: int):
    i = 0
    gains = []
    while i < len(bars) - window - 2:
        current = bars[i]
        execution_price = current.close * (1 - threshold)
        execution_bar = bars[i + 1]
        if execution_price > execution_bar.low:
            window_high = max(bar.high for bar in bars[i + 2 : i + window + 2])
            gain = (window_high - execution_price) / execution_price
            if gain < threshold:
                gain = (bars[i + window + 1].close - execution_price) / execution_price
            else:
                gain = threshold
        else:
            gain = 0.0
        gains.append(gain)
        i += 1
    return StrategyResult(bars, gains)


def simulate_short_strategy(bars: list[Bar], threshold: float, window: int):
    i = 0
    gains = []
    while i < len(bars) - window - 1:
        current = bars[i]
        current_close = current.close
        window_low = min(bar.low for bar in bars[i + 1 : i + window + 1])
        gain = (current_close - window_low) / current_close
        if gain < threshold:
            gain = (current_close - bars[i + window].close) / current_close
        else:
            gain = threshold
        gains.append(gain)
        i += 1
    return StrategyResult(bars, gains)


def bar_size_strategy(bars: list[Bar], threshold: float, window: int):
    pass


if __name__ == "__main__":
    file_path = "alpaca_historical_data_SPY_20150101_20250505.csv"
    bars = parse_csv(file_path)
    bars = [
        bar
        for bar in bars
        if bar.dt.time() >= time(9, 45) and bar.dt.time() < time(15, 0)
    ]
    bars.sort(key=lambda x: x.dt)
    grouped = groupby(bars, lambda x: x.dt.date())
    # odds_data = []
    # for (date, group) in grouped:
    #     print(f"date: {date}")
    #     bars = list(group)
    #     odds = calculate_odds(bars, 0.0002, 20)
    #     odds_data.append(odds)
    #     print(f"odds: {odds}")
    # plt.plot(odds_data)
    compounded_gains = [1.0]
    sum_gains = [0]
    for date, group in grouped:
        print(f"date: {date}")
        daily_bars = list(group)
        gain = simulate_long_strategy(daily_bars, 0.005, 30)
        compounded_gains.append((1 + gain.sum_gains) * compounded_gains[-1])
        sum_gains.append(gain.sum_gains + sum_gains[-1])
        print(f"gain: {gain.compounded_results}")

    _, axs = plt.subplots(3)
    axs[0].plot(compounded_gains)
    axs[0].set_yscale("log")
    axs[1].plot(sum_gains)
    axs[2].plot([bar.wap for bar in bars])
    plt.show()
