#!/usr/bin/env python3
"""Deterministic synthetic stress scenarios; NOT model training or real backtesting."""
import json,random,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
REGIMES={
 'bull_trend':(.002,.015,0),'bear_trend':(-.002,.015,0),
 'sideways':(0,.012,0),'high_volatility':(0,.06,0),
 'flash_crash':(0,.02,-.32),'pump_and_dump':(0,.025,.35),
 'slow_recovery':(.001,.022,-.18),'whipsaw':(0,.03,0),
 'liquidity_shock':(0,.045,-.24),'low_volatility':(0,.004,0),
 'trend_reversal':(.002,.018,0),'gap_risk':(0,.016,-.22)
}
def simulate(name,seed,n=240):
 rng=random.Random(seed);mu,vol,shock=REGIMES[name];p=100.;prices=[p]
 for i in range(n):
  drift=mu
  if name=='trend_reversal' and i>=n//2:drift=-mu
  ret=drift+rng.gauss(0,vol)
  if name=='whipsaw':ret+=.028*(1 if i%4<2 else -1)
  if name=='pump_and_dump' and i==n//2:ret+=shock
  if name=='pump_and_dump' and i==n//2+2:ret-=.32
  if name in ('flash_crash','liquidity_shock','slow_recovery','gap_risk') and i==n//2:ret+=shock
  if name=='slow_recovery' and i>n//2:ret+=.001
  p=max(.01,p*(1+max(-.85,ret)));prices.append(p)
 return prices
def strategy(prices,model,cost):
 cash=1000.;units=0.;orders=0;peak=1000.;dd=0.;signal=None
 for i in range(2,len(prices)):
  p=prices[i]
  # Decisions use only prior observations; fills at following point.
  if signal=='buy' and units==0:
   budget=cash*.10;units=budget/(p*(1+cost));cash-=budget;orders+=1
  elif signal=='sell' and units:
   cash+=units*p*(1-cost);units=0;orders+=1
  signal=None
  past=prices[i-1];previous=prices[i-2]
  if model=='momentum':direction=past>previous
  elif model=='reversion':direction=past<previous
  elif model=='trend':direction=past>sum(prices[max(0,i-15):i])/len(prices[max(0,i-15):i])
  else:direction=False
  if units==0 and direction:signal='buy'
  if units and not direction:signal='sell'
  equity=cash+units*p*(1-cost)
  peak=max(peak,equity);dd=max(dd,(peak-equity)/peak)
 final=cash+units*prices[-1]*(1-cost)
 return {'return_pct':round((final/1000-1)*100,3),'max_drawdown_pct':round(dd*100,3),'orders':orders}
results=[]
for regime in REGIMES:
 for seed in range(30):
  prices=simulate(regime,seed)
  for cost in (.0035,.01,.03):
   for model in ('momentum','reversion','trend','cash'):
    results.append({'scenario':regime,'seed':seed,'cost_per_fill':cost,'strategy':model,**strategy(prices,model,cost)})
summary=[]
for regime in REGIMES:
 for model in ('momentum','reversion','trend','cash'):
  xs=[x for x in results if x['scenario']==regime and x['strategy']==model and x['cost_per_fill']==.0035]
  summary.append({'scenario':regime,'strategy':model,'mean_return_pct':round(sum(x['return_pct'] for x in xs)/len(xs),3),'worst_return_pct':min(x['return_pct'] for x in xs),'worst_drawdown_pct':max(x['max_drawdown_pct'] for x in xs),'runs':len(xs)})
out={'disclaimer':'SYNTHETIC deterministic scenarios, not historical data, trained AI, or proof of profitability. Hypothetical next-observation fills, no market impact or leverage. Synthetic distribution assumptions can miss real-world risks.','regimes':list(REGIMES),'seeds_per_regime':30,'costs_per_fill':[.0035,.01,.03],'strategies':['momentum','reversion','trend','cash'],'total_runs':len(results),'summary':summary}
(ROOT/'scenario-report.json').write_text(json.dumps(out,indent=2)+'\n')
print('Scenario tests:',len(results),'across',len(REGIMES),'regimes')
