from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.data.models.trades import Trade
from zoneinfo import ZoneInfo
import csv
import duckdb
import tempfile
from datetime import datetime

con = duckdb.connect("db.db")

client = StockHistoricalDataClient(
    "AKC34VO0N5IFT6H523PT", "Lkl6avOAY53AMWMcFnvoi3CAKLMfDNf3Yix3N0Xt"
)


def setup_database():
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_trades (
            symbol TEXT,
            ts TIMESTAMP,
            price DOUBLE,
            size INTEGER,
            exchange TEXT,
            tape TEXT,
            conditions TEXT[]
        )
        """
    )


def insert_trades_into_db(filename: str):
    con.execute(
        f"""
            COPY stock_trades FROM '{filename}' (AUTO_DETECT TRUE, HEADER TRUE);
        """
    )


def write_trades_to_csv(trades: list[Trade], filename: str):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["symbol", "ts", "price", "size", "exchange", "conditions", "tape"]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.timestamp,
                    trade.price,
                    trade.size,
                    trade.exchange,
                    trade.tape,
                    f"[{",".join(trade.conditions)}]",
                ]
            )


def download_historical_trades(
    symbol: str, start: datetime, end: datetime
) -> list[Trade]:
    request = StockTradesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
    )
    return client.get_stock_trades(request)[symbol]


if __name__ == "__main__":
    setup_database()
    symbols = [
        "SPY",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "BRK.B",
        "JPM",
        "NFLX",
    ]
    for symbol in symbols:
        print(symbol)
        start = datetime(2025, 12, 25, tzinfo=ZoneInfo("America/New_York"))
        end = datetime(2025, 12, 27, tzinfo=ZoneInfo("America/New_York"))
        trades = download_historical_trades(symbol, start, end)
        with tempfile.NamedTemporaryFile() as f:
            print(f.name)
            write_trades_to_csv(trades, f.name)
            insert_trades_into_db(f.name)
