from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, date
from alpaca.data.models.bars import Bar
from alpaca.data.models.trades import Trade
from zoneinfo import ZoneInfo
import csv
import os
import gzip
from multiprocessing import Pool


client = StockHistoricalDataClient(
    "AKC34VO0N5IFT6H523PT", "Lkl6avOAY53AMWMcFnvoi3CAKLMfDNf3Yix3N0Xt"
)


def download_historical_data(symbol: str, start: datetime, end: datetime) -> list[Bar]:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    return client.get_stock_bars(request)[symbol]


def download_historical_trades(
    symbol: str, start: datetime, end: datetime
) -> list[Trade]:
    request = StockTradesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
    )
    return client.get_stock_trades(request)[symbol]


def save_historical_data_to_csv(
    symbol: str, start: datetime, end: datetime, filename: str
):
    bars = download_historical_data(symbol, start, end)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "WAP"])
        for bar in bars:
            writer.writerow(
                [
                    bar.timestamp.astimezone(ZoneInfo("America/New_York")),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.vwap,
                ]
            )


def _save_historical_data_to_csv_helper(symbol: str, dt: datetime):
    print(f"Downloading trades for {symbol} on {dt}")
    try:
        filename = f"trade_data/{symbol}/{dt.strftime('%Y%m%d')}.csv"
        if os.path.exists(f"{filename}.gz"):
            print(f"File {filename}.gz already exists, skipping download.")
            return
        trades = download_historical_trades(
            symbol,
            dt,
            dt + timedelta(days=1) - timedelta(microseconds=1),
        )
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Price", "Size"])
            for trade in trades:
                writer.writerow(
                    [
                        trade.timestamp.astimezone(ZoneInfo("America/New_York")),
                        trade.price,
                        trade.size,
                    ]
                )
        with open(filename, "rb") as f_in:
            with gzip.open(f"{filename}.gz", "wb") as f_out:
                f_out.writelines(f_in)
        os.remove(filename)
    except Exception as e:
        print(f"Error downloading trades for {symbol} on {dt}: {e}")


def save_historical_trades_to_csv(symbol: str, start: date, end: date, filename: str):
    csv_folder = f"trade_data/{symbol}"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    current = start
    arr: list[date] = []
    while current <= end:
        arr.append(current)
        current += timedelta(days=1)
    with Pool(4) as pool:
        pool.starmap(
            _save_historical_data_to_csv_helper,
            [(symbol, dt) for dt in arr],
        )


if __name__ == "__main__":
    symbols = ["SPY"]
    for symbol in symbols:
        start = datetime(2020, 3, 25, tzinfo=ZoneInfo("America/New_York"))
        end = datetime.now(ZoneInfo("America/New_York"))
        save_historical_trades_to_csv(symbol, start.date(), end.date(), symbol)
        # filename = f"alpaca_historical_data_{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        # save_historical_data_to_csv(symbol, start, end, filename)
