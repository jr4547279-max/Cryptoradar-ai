#!/usr/bin/env python3
"""Deterministic checks; synthetic fixtures are NEVER model training data."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from regime_diagnostics import classify,enrich,regime_diagnostics,selective_diagnostics
from historical_research import walk_forward
def run():
    assert classify(.09,.02)=='uptrend'
    assert classify(-.09,.02)=='downtrend'
    assert classify(.0,.07)=='high_volatility'
    assert classify(.0,.02)=='range_or_mixed'
    candles={'BTCUSDT':[(i*86400000,100*(1.002**i)) for i in range(870)]}
    labels=enrich(candles)
    assert len(labels)==856
    assert ('BTCUSDT',14*86400000) in labels
    rows=[(i*86400000,'BTCUSDT',[.002,.006,.014,.028],int(i%3==0)) for i in range(14,869)]
    diag=regime_diagnostics(rows,candles)
    assert sum(diag['counts'].values())==len(rows)
    # Synthetic fixture checks the evaluation code executes; results have no predictive meaning.
    result=walk_forward(rows)
    assert result['status']=='evaluated',result
    for fold in result['folds']:
        assert fold['trainExamples']>0 and fold['testExamples']>0
        assert set(fold['brier'])=={'constant','momentum','reversal','logistic'}
    selective=selective_diagnostics(rows,result['folds'])
    assert selective['status']=='evaluated'
    assert selective['selected']<=selective['total']
    print('Research diagnostic tests passed (synthetic fixtures; not model training).')
if __name__=='__main__':run()
