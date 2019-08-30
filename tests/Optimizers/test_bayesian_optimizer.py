import os
from quantfolio import Portfolio
from quantfolio.Optimizers import bayesianOptimizer

cwd = os.path.dirname(os.path.realpath(__file__))

def test_bayesian_optimizer():
    p = Portfolio()
    p.addAsset(cwd + "/../fixtures/A35.csv", "A35")
    p.addAsset(cwd + "/../fixtures/BAB.csv", "BAB")
    p.addAsset(cwd + "/../fixtures/IWDA.csv", "IWDA")
    p.generateReturnsDataframe()
    p.rf = 0.02
    results = bayesianOptimizer(p, sims=3)
    optimisedWeights = results.optimisedWeights()

    assert "Sharpe" in optimisedWeights
    assert "Returns" in optimisedWeights
    assert "Std" in optimisedWeights
    assert "A35" in optimisedWeights
    assert "BAB" in optimisedWeights
    assert "IWDA" in optimisedWeights
    
    assert results.results.shape[0] == 3

