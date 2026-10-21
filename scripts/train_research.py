#!/usr/bin/env python3
"""V18 chronological research training. Standard-library logistic regression; never trades.
Runs only after sufficient real, timestamped data exists. No synthetic data in training.
"""
import json, pathlib, math, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
raw=json.loads((ROOT/'research-data.json').read_text())
obs=raw.get('observations',[])
by_coin={}
for x in obs:
 try:
  if math.isfinite(float(x['price'])) and float(x['price'])>0:
   by_coin.setdefault(x['id'],[]).append((int(x['time']),float(x['price'])))
 except (KeyError,TypeError,ValueError):pass
for v in by_coin.values():v.sort()
H=3600000; DAY=24*H
examples=[]
for coin,points in by_coin.items():
 if len(points)<5:continue
 # One example per UTC day per coin; never reuse the same daily close repeatedly.
 daily={}
 for t,p in points:daily[t//DAY]=(t,p)
 ordered=sorted(daily.values())
 for j,(t,p) in enumerate(ordered):
  def prior(age):
   candidates=[(tt,pp) for tt,pp in ordered[:j] if tt<=t-age]
   return candidates[-1][1] if candidates else None
  a=prior(DAY);b=prior(2*DAY);c=prior(3*DAY)
  future=[(tt,pp) for tt,pp in ordered[j+1:] if tt>=t+DAY]
  if not (a and b and c and future):continue
  y=int(future[0][1]>p)
  # Features available at forecast creation only, with capped log returns.
  x=[max(-.5,min(.5,math.log(p/a))),max(-.5,min(.5,math.log(a/b))),max(-.5,min(.5,math.log(b/c)))]
  examples.append((t,coin,x,y))
examples.sort(key=lambda z:z[0])
result={'generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'insufficient_data','model':'regularized_logistic_regression',
 'trainingData':'Real recorded CoinGecko prices only; no synthetic scenarios',
 'validation':'Chronological 70% train / 30% later holdout with a 24-hour embargo; no random shuffling',
 'examples':len(examples),'uniqueCoins':len({x[1] for x in examples}),
 'warning':'Educational research, not a trading recommendation or proof of future predictive skill.'}
# Avoid training on very short or highly correlated recent samples.
if len(examples)>=300 and len({x[0]//DAY for x in examples})>=21 and len({x[1] for x in examples})>=10:
 split=int(len(examples)*.7);boundary=examples[split][0]
 train=[r for r in examples if r[0]<boundary-DAY]
 test=[r for r in examples if r[0]>=boundary]
 if len(train)>=150 and len(test)>=50 and len({r[0]//DAY for r in test})>=5:
  weights=[0.,0.,0.,0.]
  for epoch in range(300):
   grad=[0.]*4
   for _,_,x,y in train:
    v=max(-25.,min(25.,weights[0]+sum(weights[i+1]*x[i] for i in range(3))))
    pred=1/(1+math.exp(-v))
    for k,z in enumerate([1.]+x):grad[k]+=(pred-y)*z
   rate=.25
   for k in range(4):
    weights[k]-=rate*(grad[k]/len(train)+(.005*weights[k] if k else 0))
  def brier(data,baseline=False):
   return sum(((sum(y for *_,y in train)/len(train)) if baseline else 1/(1+math.exp(-max(-25,min(25,weights[0]+sum(weights[i+1]*x[i] for i in range(3))))))-y)**2 for _,_,x,y in data)/len(data)
  model_score=brier(test);baseline_score=brier(test,True)
  result.update({'status':'evaluated','trainExamples':len(train),'testExamples':len(test),
    'testDays':len({r[0]//DAY for r in test}),
    'holdoutBrier':round(model_score,5),'constantTrainRateBrier':round(baseline_score,5),
    'betterThanConstantBaseline':model_score<baseline_score,
    'note':'A single evaluation is not sufficient to establish predictive skill. Repeated coins share market shocks.'})
else:
 result['note']='Waiting for at least 21 distinct UTC days, 300 examples and 10 coins before training.'
(ROOT/'training-report.json').write_text(json.dumps(result,indent=2)+'\n')
print('V18 training:',result['status'],'examples:',len(examples))
