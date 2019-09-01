import math
import datetime
import os
import pandas as pd
from quantfolio import Portfolio, removeNonTradingDays, forwardFillPrices

cwd = os.path.dirname(os.path.realpath(__file__))

# TODO Test if backtest if creating orders
def testBacktest():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/A35.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/BAB.csv", "Asset2")

    results, plot = p.backtest([0.5, 0.5])


def testAddAsset():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")

    assert p.assetNames == ["Asset1"]
    assert p.assetDatas[0].iloc[0]["Adj Close"] == 15.996025
    assert p.assetReturns[0].iloc[1]["Adj Close"] == math.log(
        p.assetDatas[0].iloc[1]["Adj Close"] / p.assetDatas[0].iloc[0]["Adj Close"]
    )


def testAddExchangeQuoted():
    p = Portfolio()
    p.addExchangeRate(cwd + "/fixtures/USDSGD.csv", "USD")

    assert (
        p.exchange["USD"].iloc[0]["Close"] == 1.2867
    ), "Closing price should not be inverse"
    assert not p.exchange["USD"].iloc[1]["Close"] == None, "NA values should be filled"


def testAddExchangeBase():
    # base = True if portfolio currency is the base pair, ie SGD in SGD/EUR
    p = Portfolio()
    p.addExchangeRate(cwd + "/fixtures/SGDEUR.csv", "EUR", True)

    assert (
        p.exchange["EUR"].iloc[0]["Close"] == 2.081078831266128
    ), "Closing price should be inverse"
    assert not p.exchange["EUR"].iloc[1]["Close"] == None, "NA values should be filled"


def testExchangeAdjustment():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addExchangeRate(cwd + "/fixtures/USDSGD.csv", "USD")
    p.exchangeAdjustment(0, "USD")

    assert p.assetDatas[0].iloc[0]["Adj Close"] == 20.864415208749996
    assert p.assetReturns[0].iloc[-1]["Adj Close"] == 2.2565439772970107e-05
    assert not p.assetDatas[0].isnull().values.any(), "NaN is introduced in data"
    assert (
        not p.assetReturns[0].iloc[1:-1].isnull().values.any()
    ), "NaN is introduced in returns"


def testCommonInterval():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset2")
    fromDate, toDate = p.commonInterval()

    assert fromDate == datetime.datetime.fromisoformat("2009-11-25")
    assert toDate == datetime.datetime.fromisoformat("2009-12-09")


def testGenerateReturnsDataframe():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset2")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset3")
    p.generateReturnsDataframe()

    assert all(
        [
            a == b
            for a, b in zip(p.assetReturnsDf.columns, ["Asset1", "Asset2", "Asset3"])
        ]
    )


def testPortfolioReturns():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset2")
    assert p.portfolioReturns([0.5, 0.5]) == -0.2504711195035464


def testPortfolioStd():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset2")
    assert p.portfolioStd([0.5, 0.5]) == 0.061536653450580125


def testPortfolioPerformance():
    p = Portfolio()
    p.addAsset(cwd + "/fixtures/Asset.csv", "Asset1")
    p.addAsset(cwd + "/fixtures/Asset2.csv", "Asset2")
    perf = p.portfolioPerformance([0.5, 0.5])
    assert perf == {
        "returns": -0.2504711195035464,
        "std": 0.061536653450580125,
        "sharpe": -4.070275282433077,
    }

def testRemoveNonTradingDays():
    data = pd.read_csv(cwd + "/fixtures/USDSGD.csv", index_col="Date", parse_dates=True)
    assert data.index.contains("2009-11-15")
    assert not removeNonTradingDays(data).index.contains("2009-11-15")


def testForwardFillPrices():
    data = pd.read_csv(cwd + "/fixtures/USDSGD.csv", index_col="Date", parse_dates=True)
    assert math.isnan(data.loc["2009-11-12"]["Open"])
    assert not math.isnan(forwardFillPrices(data).loc["2009-11-12"]["Open"])