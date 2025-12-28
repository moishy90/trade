import csv
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Bar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    wap: float


def parse_csv(file_path: str) -> list[Bar]:
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        bars = []
        for row in reader:
            dt = datetime.fromisoformat(row["Date"])
            open_price = float(row["Open"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            close_price = float(row["Close"])
            volume = float(row["Volume"])
            wap = float(row["WAP"])
            bars.append(
                Bar(dt, open_price, high_price, low_price, close_price, volume, wap)
            )
    return bars
