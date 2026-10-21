"""Score existing timestamped GBP forward forecasts without changing predictions."""
import json, pathlib
from reward_referee import referee

ROOT=pathlib.Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'research-data.json').read_text())
forecasts=source.get('forecasts',[])
agents=('momentum','mean_reversion','trend_48h','neutral','council')
results={}
for agent in agents:
    records=[]
    for f in forecasts:
        if f.get('outcome') not in (0,1) or not isinstance(f.get('models'),dict):continue
        m=f['models']
        if not all(isinstance(m.get(n),(int,float)) for n in agents[:-1]):continue
        p=sum(m[n] for n in agents[:-1])/4 if agent=='council' else m[agent]
        records.append({'asset':f.get('id'),'created':f.get('created'),
                        'resolved':f.get('scoredAt'),'probability':p,'baseline':.5,
                        'outcome':f['outcome'],'abstain':False})
    results[agent]=referee(records)
report={'status':'SHADOW_RESEARCH','source':'research-data.json existing GBP forward forecasts',
        'modelFamily':'V16 fixed untrained research rules (not historical trained council)',
        'agents':results,
        'warning':'Exploratory reward score; forecast file is mutable and is not independently immutable. No reward-based training or trading authorization.'}
(ROOT/'reward-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Shadow reward report:',{k:v['status'] for k,v in results.items()})
