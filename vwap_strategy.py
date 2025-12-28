import pandas
from datetime import datetime, date, time
from matplotlib import pyplot as plt
from zoneinfo import ZoneInfo
from util import parse_csv
import math
import signal
from util import Bar
import numpy as np
from dataclasses import dataclass
from enum import Enum
from matplotlib.widgets import MultiCursor
import mplcursors

signal.signal(signal.SIGINT, signal.SIG_DFL)


@dataclass
class VWAPStrategyData:
    bar: Bar
    vwap_std: float
    vwap: float


@dataclass
class STDThreshold:
    std_threshold: float


@dataclass
class PercentThreshold:
    percent_threshold: float


class VWAPStrategy:
    def __init__(
        self,
        bars_skip_begin_count: int,
        bars_skip_end_count: int,
        threshold: STDThreshold | PercentThreshold,
        std_multiple: float,
    ) -> None:
        self.bars_skip_begin_count = bars_skip_begin_count
        self.bars_skip_end_count = bars_skip_end_count
        self.threshold = threshold
        self.std_multiple = std_multiple
        self.gains: list[float] = [1.0]

    def run_strategy(self, data: list[VWAPStrategyData]):
        i = self.bars_skip_begin_count
        mins_maxs = self._construct_mins_maxs(data)
        closing_bar = data[-1].bar
        gains = 0.0
        result = []
        while i < len(data) - self.bars_skip_end_count:
            bar = data[i].bar
            vwap_std = data[i].vwap_std
            vwap = data[i].vwap
            if bar.high > (vwap + vwap_std * self.std_multiple):
                sell_price = vwap + vwap_std * self.std_multiple
                print(f"sell short price is {sell_price}")
                match self.threshold:
                    case PercentThreshold(percent_threshold):
                        mn, _ = mins_maxs[i + 2]
                        if mn <= sell_price * (1 - percent_threshold):
                            gains += percent_threshold
                        else:
                            gains += (sell_price - closing_bar.high) / sell_price
                    case STDThreshold(std_threshold):
                        j = i + 1
                        while j < len(data):
                            buy_price = data[j].vwap + data[j].vwap_std * std_threshold
                            if data[j].bar.low < buy_price:
                                print(f"buy to close price is {buy_price}")
                                gains += (sell_price - buy_price) / sell_price
                                break
                            j += 1
                        if j == len(data):
                            gains += (sell_price - closing_bar.high) / sell_price
                result.append((sell_price, "sell", i + 1, data[i + 1]))
            elif bar.low < (vwap - vwap_std * self.std_multiple):
                purchase_price = vwap - vwap_std * self.std_multiple
                print(f"purchase long price is {purchase_price}")
                match self.threshold:
                    case PercentThreshold(percent_threshold):
                        _, mx = mins_maxs[i + 2]
                        if mx >= purchase_price * (1 + percent_threshold):
                            gains += percent_threshold
                        else:
                            gains += (closing_bar.low - purchase_price) / purchase_price
                    case STDThreshold(std_threshold):
                        j = i + 1
                        while j < len(data):
                            sell_price = data[j].vwap - data[j].vwap_std * std_threshold
                            if data[j].bar.high > sell_price:
                                gains += (sell_price - purchase_price) / sell_price
                                print(f"sell to close price is {sell_price}")
                                break
                            j += 1
                        if j == len(data):
                            gains += (closing_bar.low - purchase_price) / purchase_price
                print(gains)
                result.append((purchase_price, "buy", i + 1, data[i + 1]))
            i += 1
        self.gains.append(self.gains[-1] * (1 + gains))
        return result

    def plot_gains(self):
        plt.plot(self.gains)
        plt.show()

    def _construct_mins_maxs(self, data: list[VWAPStrategyData]):
        mn = float("inf")
        mx = float("-inf")
        result: list[tuple[float, float]] = []
        i = len(data) - 1
        while i >= 0:
            low = data[i].bar.low
            high = data[i].bar.high
            if low < mn:
                mn = low
            if high > mx:
                mx = high
            result.append((mn, mx))
            i -= 1
        result.reverse()
        return result


df = pandas.read_csv("alpaca_historical_data_AAOI_20150101_20250513.csv")
df["dt"] = pandas.to_datetime(df["Date"], utc=True).dt.tz_convert("America/New_York")


def get_dataframe_between_dates(df: pandas.DataFrame, start: date, end: date):
    return df.loc[(df["dt"] >= start) & (df["dt"] <= end)]


def filter_for_market_hours(df: pandas.DataFrame):
    return df.loc[
        (df["dt"].dt.time >= time(hour=9, minute=30))
        & (df["dt"].dt.time < time(hour=16, minute=0))
    ]


df = get_dataframe_between_dates(
    df,
    datetime(year=2024, month=8, day=12, tzinfo=ZoneInfo("America/New_York")),
    datetime(
        year=2025,
        month=5,
        day=2,
        hour=23,
        minute=59,
        second=59,
        tzinfo=ZoneInfo("America/New_York"),
    ),
)


def add_windowed_std_columns(df: pandas.DataFrame, window_size: int):
    df["volumeVwapSum"] = (df["Volume"] * df["WAP"]).rolling(window_size).sum()
    df["volumeSum"] = df["Volume"].rolling(window_size).sum()
    df["price"] = df["volumeVwapSum"] / df["volumeSum"]
    df["volumeVwap2Sum"] = (
        (df["Volume"] * df["WAP"] * df["WAP"]).rolling(window_size).sum()
    )
    df["zero"] = 0
    df["variance"] = (df["volumeVwap2Sum"] / df["volumeSum"]) - (df["price"] ** 2)
    df["variance"] = np.maximum(df["variance"], df["zero"])
    df["std"] = df["variance"].apply(math.sqrt)
    df["VWAP"] = df["price"]
    # df["std"] = df["VWAP"].expanding().std(ddof=0)
    df["std1+"] = df["VWAP"] + df["std"]
    df["std1-"] = df["VWAP"] - df["std"]
    df["std2+"] = df["VWAP"] + 2 * df["std"]
    df["std2-"] = df["VWAP"] - 2 * df["std"]
    df["std3+"] = df["VWAP"] + 3 * df["std"]
    df["std3-"] = df["VWAP"] - 3 * df["std"]
    df["volume_std"] = df["Volume"].rolling(window_size).std(ddof=0)
    df["volume_mean"] = df["Volume"].rolling(window_size).mean()
    return df


def add_std_columns(df: pandas.DataFrame):
    """
    VWAP Calculation based on thinkorswim VWAP code.
    #
    # Charles Schwab & Co. (c) 2011-2025
    #

    input numDevDn = -2.0;
    input numDevUp = 2.0;
    input timeFrame = {default DAY, WEEK, MONTH};

    def cap = getAggregationPeriod();
    def errorInAggregation =
        timeFrame == timeFrame.DAY and cap >= AggregationPeriod.WEEK or
        timeFrame == timeFrame.WEEK and cap >= AggregationPeriod.MONTH;
    assert(!errorInAggregation, "timeFrame should be not less than current chart aggregation period");

    def yyyyMmDd = getYyyyMmDd();
    def periodIndx;
    switch (timeFrame) {
    case DAY:
        periodIndx = yyyyMmDd;
    case WEEK:
        periodIndx = Floor((daysFromDate(first(yyyyMmDd)) + getDayOfWeek(first(yyyyMmDd))) / 7);
    case MONTH:
        periodIndx = roundDown(yyyyMmDd / 100, 0);
    }
    def isPeriodRolled = compoundValue(1, periodIndx != periodIndx[1], yes);

    def volumeSum;
    def volumeVwapSum;
    def volumeVwap2Sum;

    if (isPeriodRolled) {
        volumeSum = volume;
        volumeVwapSum = volume * vwap;
        volumeVwap2Sum = volume * Sqr(vwap);
    } else {
        volumeSum = compoundValue(1, volumeSum[1] + volume, volume);
        volumeVwapSum = compoundValue(1, volumeVwapSum[1] + volume * vwap, volume * vwap);
        volumeVwap2Sum = compoundValue(1, volumeVwap2Sum[1] + volume * Sqr(vwap), volume * Sqr(vwap));
    }
    def price = volumeVwapSum / volumeSum;
    def deviation = Sqrt(Max(volumeVwap2Sum / volumeSum - Sqr(price), 0));

    plot VWAP = price;
    plot UpperBand = price + numDevUp * deviation;
    plot LowerBand = price + numDevDn * deviation;

    VWAP.setDefaultColor(getColor(0));
    UpperBand.setDefaultColor(getColor(2));
    LowerBand.setDefaultColor(getColor(4));
    """
    df["volumeVwapSum"] = (df["Volume"] * df["WAP"]).cumsum()
    df["volumeSum"] = df["Volume"].cumsum()
    df["price"] = df["volumeVwapSum"] / df["volumeSum"]
    df["volumeVwap2Sum"] = (df["Volume"] * df["WAP"] * df["WAP"]).cumsum()
    df["zero"] = 0
    df["variance"] = (df["volumeVwap2Sum"] / df["volumeSum"]) - (df["price"] ** 2)
    df["variance"] = np.maximum(df["variance"], df["zero"])
    df["std"] = df["variance"].apply(math.sqrt)
    df["VWAP"] = df["price"]
    df["moving_average"] = df["WAP"].rolling(60).mean()
    # df["std"] = df["VWAP"].expanding().std(ddof=0)
    df["std1+"] = df["VWAP"] + df["std"]
    df["std1-"] = df["VWAP"] - df["std"]
    df["std2+"] = df["VWAP"] + 2 * df["std"]
    df["std2-"] = df["VWAP"] - 2 * df["std"]
    df["std3+"] = df["VWAP"] + 3 * df["std"]
    df["std3-"] = df["VWAP"] - 3 * df["std"]
    df["volume_std"] = df["Volume"].expanding().std(ddof=0)
    df["volume_mean"] = df["Volume"].expanding().mean()
    return df


def plot(df: pandas.DataFrame):
    plt.style.use("dark_background")

    fig, axs = plt.subplots(3)

    axs[0].plot(df["WAP"].to_numpy(), label="WAP (Price)")
    axs[0].plot(df["VWAP"].to_numpy(), label="VWAP")
    axs[0].plot(df["std1+"].to_numpy(), label="std1+")
    axs[0].plot(df["std1-"].to_numpy(), label="std1-")
    axs[0].plot(df["std2+"].to_numpy(), label="std2+")
    axs[0].plot(df["std2-"].to_numpy(), label="std2-")
    axs[0].plot(df["std3+"].to_numpy(), label="std3+")
    axs[0].plot(df["std3-"].to_numpy(), label="std3-")
    axs[0].plot(df["moving_average"].to_numpy(), label="moving_average")
    # axs[0].set_ylim([df["std3-"].to_numpy().min(), df["std3+"].to_numpy().max()])
    axs[1].plot(df["std"].to_numpy(), label="std")
    # axs[1].set_ylim([df["std"].to_numpy().min(), df["std"].to_numpy().max()])
    # axs[2].set_yscale("log")
    axs[2].plot(
        df["Volume"].to_numpy(),
        label="Volume",
    )
    axs[2].plot(df["volume_std"].to_numpy(), label="volume_std")
    axs[2].plot(df["volume_mean"].to_numpy(), label="volume_mean")
    # multi = MultiCursor(None, axes=axs, useblit=False, color="r")
    # mplcursors.cursor()
    # axs[0].legend()
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.show()


def df_to_bars(df: pandas.DataFrame):
    result = []
    for row in df.to_dict(orient="records"):
        result.append(
            Bar(
                dt=row["Date"],
                open=row["Open"],
                high=row["High"],
                low=row["Low"],
                close=row["Close"],
                volume=row["Volume"],
                wap=row["WAP"],
            )
        )
    return result


def df_to_vwap_strategy_data(df: pandas.DataFrame):
    result = []
    for row in df.to_dict(orient="records"):
        result.append(
            VWAPStrategyData(
                bar=Bar(
                    dt=row["Date"],
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                    wap=row["WAP"],
                ),
                vwap_std=row["std"],
                vwap=row["VWAP"],
            )
        )
    return result


df = filter_for_market_hours(df)
strategy = VWAPStrategy(
    bars_skip_begin_count=60,
    bars_skip_end_count=30,
    threshold=STDThreshold(-1.0),
    std_multiple=3.0,
)
df = df.loc[::-1]
for d, df in df.groupby(df["dt"].dt.date, sort=False):
    print(d)
    # df = df.loc[100:]
    # df = df.loc[::-1].copy()
    df = add_std_columns(df)
    # strategy_data = df_to_vwap_strategy_data(df)
    # report = strategy.run_strategy(strategy_data)

    # if strategy.gains[-1] < strategy.gains[-2]:
    #     print(report)
    # if report:
    plot(df)
    # plt.show()

# strategy.plot_gains()
