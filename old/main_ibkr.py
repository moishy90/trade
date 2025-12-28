from ibapi.client import *
from ibapi.wrapper import *
import time
import datetime
import threading
import csv
import asyncio
import pytz
import random


class TestApp(EClient, EWrapper):
  def __init__(self):
    EClient.__init__(self, self)
    self.events = {}
    self.data = {}
  
  def nextValidId(self, orderId):
    self.orderId = orderId
  
  def nextId(self):
    self.orderId += 1
    return self.orderId
  
  def currentTime(self, time):
    print(time)

  def error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject=""):
    print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")

  def reqHistoricalData(self, reqId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions):
    print(f"start request for {reqId}")
    self.events[reqId] = asyncio.Event()
    self.data[reqId] = []
    return super().reqHistoricalData(reqId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions)

  def historicalData(self, reqId, bar):
    print(f"reqId: {reqId}, bar: {bar}")
    self.data[reqId].append(bar)
  
  def historicalDataEnd(self, reqId, start, end):
    print(f"reqId: {reqId}, start: {start}, end: {end}")
    loop.call_soon_threadsafe(self.events[reqId].set)



def convert_to_datetime(datetime_str):
    format_str = "%Y%m%d %H:%M:%S"
    naive_datetime = datetime.datetime.strptime(datetime_str[:17], format_str)
    timezone_str = datetime_str[18:].strip()
    timzone = pytz.timezone(timezone_str)
    aware_datetime = timzone.localize(naive_datetime)
    return aware_datetime


async def batch_download_historical_data(symbol: str, duration: str, end_datetime: datetime.datetime, bar_size_setting: str) -> list[BarData]:
  mycontract = Contract()
  mycontract.symbol = symbol
  mycontract.secType = "STK"
  mycontract.exchange = "SMART"
  mycontract.currency = "USD"
  req_id = random.randint(0, 100000000)
  print(f"req_id: {req_id}")
  format_str = "%Y%m%d %H:%M:%S"
  endDateTime = end_datetime.strftime(format_str) + " US/Eastern"
  app.reqHistoricalData(req_id, mycontract, endDateTime, duration, bar_size_setting, "TRADES", 1, 1, False, [])
  print("returning")
  await app.events[req_id].wait()
  return app.data[req_id]


async def download_historical_data():
    symbols = ["TGT", "WBA"]
    bar_size_setting = "1 min"
    duration = '10 Y'
    end_datetime = datetime.datetime.now(tz=pytz.timezone("US/Eastern"))
    # endDateTime = "20210523 16:00:00 US/Eastern
    futures =  [(symbol, asyncio.create_task(batch_download_historical_data(symbol=symbol, duration=duration, end_datetime=end_datetime, bar_size_setting=bar_size_setting))) for symbol in symbols]
    await asyncio.gather(*[future for _, future in futures])
    for symbol, future in futures:
      data = await future
      rows = sorted(
        [[convert_to_datetime(bar.date), bar.open, bar.high, bar.low, bar.close, bar.volume, bar.wap] for bar in data],
          key=lambda x: x[0]
      )
      with open(f"historical_data_{symbol}_{'_'.join(bar_size_setting.split(' '))}.csv", "w", newline='') as f:
          writer = csv.writer(f)
          writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "WAP"])
          writer.writerows(rows)


if __name__ == "__main__":
    app = TestApp()
    app.connect("127.0.0.1", 7496, 0)
    threading.Thread(target=app.run, daemon=True).start()
    time.sleep(2)
    loop = asyncio.new_event_loop()
    # for i in range(0,5):
    #   print(app.nextId())
    # app.reqCurrentTime()

    # def get_historical_data
    loop.run_until_complete(download_historical_data())
