from quantfolio.Optimizers import OptimisationResults, utils


def test_optimised_weights():
    rf = 0.02
    results = OptimisationResults(["A1", "A2", "A3"], rf=0.02)
    results.addData(
        {
            "A1": 0.33,
            "A2": 0.33,
            "A3": 0.33,
            "Sharpe": utils.sharpe(0.06, 0.1, rf),
            "Std": 0.1,
            "Returns": 0.06,
        }
    )
    results.addData(
        {
            "A1": 1,
            "A2": 0,
            "A3": 0,
            "Sharpe": utils.sharpe(0.05, 0.08, rf),
            "Std": 0.08,
            "Returns": 0.05,
        }
    )
    results.addData(
        {
            "A1": 0.5,
            "A2": 0.5,
            "A3": 0,
            "Sharpe": utils.sharpe(0.055, 0.09, rf),
            "Std": 0.09,
            "Returns": 0.055,
        }
    )
    optimisedWeights = results.optimisedWeights()
    assert optimisedWeights["Sharpe"] == 0.3999999999999999
    assert optimisedWeights["Returns"] == 0.06
    assert optimisedWeights["Std"] == 0.1
    assert optimisedWeights["A1"] == 0.33
    assert optimisedWeights["A2"] == 0.33
    assert optimisedWeights["A3"] == 0.33
