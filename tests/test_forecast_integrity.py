"""Deterministic adversarial forecast integrity tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from forecast_integrity import audit

def fixture():
    return {'id':'bitcoin','created':1000,'price':100.,'pUp':.5,
            'models':{'momentum':.6,'neutral':.4},'outcome':1,
            'scoredAt':86401000,'scoredPrice':101.}

def test():
    assert audit([fixture()])['status']=='PASS'
    assert audit([fixture(),fixture()])['status']=='FAIL'
    for field,value in [('scoredAt',1001),('scoredPrice',99.),('pUp',.9),('price',-1.)]:
        f=fixture();f[field]=value
        assert audit([f])['status']=='FAIL',field
    f=fixture();f['outcome']=True
    assert audit([f])['status']=='FAIL'
    f=fixture();f['outcome']=None;f['scoredAt']=None;f.pop('scoredPrice')
    assert audit([f])['status']=='PASS'
    f=fixture()
    f['challengers']={'trend_logistic_v1':{
        'pUp':0.62,'trialId':'trend-logistic-v1-forward-001',
        'modelVersion':'trend-logistic-v1','modelHash':'a'*64,
        'trainingDatasetSha256':'b'*64}}
    assert audit([f])['status']=='PASS'
    f['challengers']['trend_logistic_v1']['pUp']=1.2
    assert audit([f])['status']=='FAIL'
    f=fixture()
    f['challengers']={'trend_logistic_v1':{
        'pUp':0.62,'trialId':'trial','modelVersion':'v1','modelHash':'x',
        'trainingDatasetSha256':'b'*64}}
    assert audit([f])['status']=='FAIL'
    print('Forward forecast integrity adversarial tests passed')

if __name__=='__main__':test()
