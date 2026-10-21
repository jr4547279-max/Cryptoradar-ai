"""Bounded overnight synthetic research. No real data or reward promotion."""
import datetime, json, pathlib, random, statistics
from scenario_stress import REGIMES, simulate, strategy

ROOT=pathlib.Path(__file__).resolve().parents[1]
END=datetime.datetime.fromisoformat('2026-10-22T09:33:49+00:00')
now=datetime.datetime.now(datetime.timezone.utc)
path=ROOT/'overnight-research.json'
if now>=END:
    print('Overnight experiment window ended; no new runs')
    raise SystemExit(0)

# Reproducible fresh seeds per UTC hour, separated from any forward evaluation.
batch=int(now.timestamp()//3600)
models=('momentum','reversion','trend','cash')
costs=(.0035,.01,.03)
rng=random.Random(batch)
totals={m:[] for m in models}
worst={m:0 for m in models}
count=0
for regime in REGIMES:
    for j in range(100):
        seed=rng.randrange(1,2**31)
        prices=simulate(regime,seed,n=240)
        for cost in costs:
            for model in models:
                r=strategy(prices,model,cost)
                totals[model].append(r['return_pct'])
                worst[model]=max(worst[model],r['max_drawdown_pct'])
                count+=1
summary={m:{'meanReturnPct':round(statistics.mean(totals[m]),4),
            'worstDrawdownPct':round(worst[m],3),
            'samples':len(totals[m])} for m in models}
try: state=json.loads(path.read_text())
except (OSError,ValueError):state={'batches':[]}
if not any(x.get('hour')==batch for x in state['batches']):
    state['batches'].append({'hour':batch,'at':now.isoformat(),
        'simulations':count,'regimes':len(REGIMES),'seedPolicy':'fixed by UTC hour',
        'strategies':summary})
state['totalSimulations']=sum(x['simulations'] for x in state['batches'])
state['endAt']=END.isoformat()
state['status']='SYNTHETIC_RESEARCH_ONLY'
state['disclaimer']='Stress tests on artificial price paths; no training, agent rewards, forward validation, or evidence of profit. No models are promoted.'
path.write_text(json.dumps(state,indent=2)+'\n')
print('Overnight batch:',count,'synthetic scenario-strategy-cost evaluations; cumulative',state['totalSimulations'])
