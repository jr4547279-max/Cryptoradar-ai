"""Adversarial checks for experimental reward referee."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reward_referee import referee

def records(p=.8,y=1,abstain=False):
    return [{'asset':'BTC','created':i*86400000,'resolved':(i+1)*86400000,
             'probability':p,'baseline':.5,'outcome':y,'abstain':abstain} for i in range(40)]

def test():
    assert referee(records())['eligible']
    assert not referee(records(abstain=True))['eligible']
    assert not referee(records(p=.5))['eligible']
    assert not referee(records(p=.99,y=0))['eligible']
    duplicated=records();duplicated[1]=dict(duplicated[0])
    assert referee(duplicated)['status']=='INVALID_DUPLICATE'
    future=records();future[0]['resolved']=future[0]['created']
    assert referee(future)['status']=='INVALID'
    missing=records();missing[0]['probability']=float('nan')
    assert referee(missing)['status']=='INVALID'
    assert not referee(records()[:3])['eligible']
    print('Reward referee adversarial tests passed')

if __name__=='__main__':test()
