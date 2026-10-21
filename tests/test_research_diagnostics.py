#!/usr/bin/env python3
"""Deterministic checks; synthetic fixtures are NEVER model training data."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from regime_diagnostics import classify,enrich,regime_diagnostics,selective_diagnostics
from historical_research import walk_forward
from agent_council import council
from council_evaluation import evaluate_council
from trainable_council import assess as assess_trainable_council
from council_robustness import robustness
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
    assert council([float('nan'),0,0,0])['decision']=='NO_TRADE'
    assert council([0,0,0,0],estimated_roundtrip_cost=.02)['decision']=='NO_TRADE'
    assert council([.1,0,.24,.25])['decision']=='NO_TRADE'
    report=evaluate_council(rows,candles,result['folds'])
    assert report['status']=='evaluated'
    assert report['signals']<=report['observations']
    assert report['coverage']<=1
    assert all(f['signals']<=f['observations'] for f in report['folds'])
    trained=assess_trainable_council(rows)
    assert trained['status']=='evaluated',trained
    assert trained['trainExamples']>trained['testExamples']
    assert 'equal_weight_council' in trained['scores']
    assert 'baseline' in trained['scores']
    assert all(0<=v['brier']<=1 for v in trained['scores'].values())
    assert sum(b['count'] for b in trained['calibration'])==trained['testExamples']
    robust=robustness(rows)
    assert robust['status']=='evaluated',robust
    assert len(robust['folds'])==3
    assert all('BTCUSDT' in fold['perAsset'] for fold in robust['folds'])
    assert robust['testExamples']==sum(fold['testExamples'] for fold in robust['folds'])
    print('Research diagnostic tests passed (synthetic fixtures; not model training).')
if __name__=='__main__':run()
