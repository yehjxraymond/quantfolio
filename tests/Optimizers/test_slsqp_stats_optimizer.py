import os
from quantfolio import Portfolio
from quantfolio.Optimizers import slsqpStatsOptimizer

cwd = os.path.dirname(os.path.realpath(__file__))


def test_slsqp_stats_optimizer():
    p = Portfolio()
    p.addAsset(cwd + "/../fixtures/A35.csv", "A35")
    p.addAsset(cwd + "/../fixtures/BAB.csv", "BAB")
    p.addAsset(cwd + "/../fixtures/IWDA.csv", "IWDA")
    p.generateReturnsDataframe()
    p.rf = 0.02
    results = slsqpStatsOptimizer(p)
    optimisedWeights = results.optimisedWeights()

    assert "Sharpe" in optimisedWeights
    assert "Returns" in optimisedWeights
    assert "Std" in optimisedWeights
    assert "A35" in optimisedWeights
    assert "BAB" in optimisedWeights
    assert "IWDA" in optimisedWeights

    # Should converge within 50 iterations
    assert results.results.shape[0] < 50

