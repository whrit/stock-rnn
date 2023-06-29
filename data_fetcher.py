import click
import os
import pandas as pd
import random
import time
import yfinance as yf

from bs4 import BeautifulSoup
from datetime import datetime

DATA_DIR = "data"
RANDOM_SLEEP_TIMES = (1, 5)

# Replace with path to your local file
SP500_LIST_PATH = "/home/bw/Projects/stock-rnn/data/constituents-financials.csv"

def _load_symbols():
    df_sp500 = pd.read_csv(SP500_LIST_PATH, on_bad_lines='warn')
    print(df_sp500.columns.tolist())  # print the column names
    print(df_sp500.head())  # print the first few rows of the dataframe
    df_sp500.sort_values('Market Cap', ascending=False, inplace=True)
    stock_symbols = df_sp500['Symbol'].unique().tolist()
    print("Loaded %d stock symbols" % len(stock_symbols))
    return stock_symbols


def fetch_prices(symbol, out_name):
    """
    Fetch daily stock prices for stock `symbol`, since 1980-01-01.

    Args:
        symbol (str): a stock abbr. symbol, like "GOOG" or "AAPL".

    Returns: a bool, whether the fetch is succeeded.
    """
    print("Fetching {} ...".format(symbol))

    try:
        data = yf.download(symbol, start='1980-01-01')
        if data.empty:
            print("Remove {} because the data set is empty.".format(out_name))
            return False

        data.to_csv(out_name)
        print("# Fetched rows: %d [%s to %s]" % (data.shape[0], data.index[-1], data.index[0]))
    except Exception as e:
        print("Failed when fetching {}. Error: {}".format(symbol, str(e)))
        return False

    # Take a rest
    sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    print("Sleeping ... %ds" % sleep_time)
    time.sleep(sleep_time)
    return True


@click.command(help="Fetch stock prices data")
@click.option('--continued', is_flag=True)
def main(continued):
    random.seed(time.time())
    num_failure = 0

    # This is S&P 500 index
    # fetch_prices('INDEXSP%3A.INX')

    symbols = _load_symbols()
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")
        if continued and os.path.exists(out_name):
            print("Fetched", sym)
            continue

        succeeded = fetch_prices(sym, out_name)
        num_failure += int(not succeeded)

        if idx % 10 == 0:
            print("# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure))


if __name__ == "__main__":
    main()
