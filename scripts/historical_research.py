#!/usr/bin/env python3
"""Experimental historical crypto research. Public Binance Vision daily spot candles (USDT quotes).
No trading, no API keys, no claims of predictive skill. Chronological train/holdout.
"""
import csv, datetime as dt, io, json, math, pathlib, statistics, urllib.request, zipfile
from regime_diagnostics import regime_diagnostics, selective_diagnostics
from agent_council import council_diagnostics
from council_evaluation import evaluate_council
from council_robustness import robustness
from trainable_council import assess as assess_trainable_council
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'historical-training-report.json'
PAIRS=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']
MONTHS=18
def month_keys():
    today=dt.datetime.now(dt.timezone.utc).date()
    y,m=today.year,today.month
    keys=[]
    for _ in range(MONTHS):
        m-=1
        if m==0:m=12;y-=1
        keys.append(f'{y:04d}-{m:02d}')
    return list(reversed(keys))
def fetch(pair,month):
    url=f'https://data.binance.vision/data/spot/monthly/klines/{pair}/1d/{pair}-1d-{month}.zip'
    req=urllib.request.Request(url,headers={'User-Agent':'CryptoRadarResearch/1.0'})
    with urllib.request.urlopen(req,timeout=25) as res:
        blob=res.read(1000000)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as fp:
            for row in csv.reader(io.TextIOWrapper(fp)):
                try:
                    timestamp=int(row[0]);price=float(row[4])
                    if timestamp>1e13:timestamp//=1000
                    if math.isfinite(price) and price>0:yield timestamp,price
                except (ValueError,IndexError):continue
def features(p,i):
    return [max(-.3,min(.3,math.log(p[i]/p[i-k]))) for k in (1,3,7,14)]
def sigmoid(v):return 1/(1+math.exp(-max(-25,min(25,v))))
def score(rows,w):
    return sum((sigmoid(w[0]+sum(w[j+1]*x[j] for j in range(4)))-y)**2 for _,_,x,y in rows) / len(rows)

def walk_forward(rows):
    """Expanding-window evaluation. No selection of winners on test folds."""
    dates=sorted(set(r[0] for r in rows))
    if len(dates)<180:return {'status':'insufficient_data'}
    folds=[]
    boundaries=[.4,.55,.7,.85,1.]
    for k in range(4):
        lo=dates[int(len(dates)*boundaries[k])]
        hi=dates[min(len(dates)-1,int(len(dates)*boundaries[k+1])-1)]
        train=[r for r in rows if r[0]<lo-2*86400000]
        test=[r for r in rows if lo<=r[0]<=hi]
        if len(train)<300 or len(test)<80:continue
        mean=[statistics.fmean(r[2][j] for r in train) for j in range(4)]
        std=[max(1e-5,statistics.pstdev(r[2][j] for r in train)) for j in range(4)]
        def scaled(r):return [(r[2][j]-mean[j])/std[j] for j in range(4)]
        w=[0.]*5
        for _ in range(150):
            grad=[0.]*5
            for r in train:
                x=scaled(r)
                p=sigmoid(w[0]+sum(w[j+1]*x[j] for j in range(4)))
                for j,z in enumerate([1.]+x):grad[j]+=(p-r[3])*z
            for j in range(5):w[j]-=.16*(grad[j]/len(train)+(.08*w[j] if j else 0))
        base=statistics.fmean(r[3] for r in train)
        totals={name:0. for name in ('constant','momentum','reversal','logistic')}
        for r in test:
            x=scaled(r);y=r[3]
            preds={'constant':base,'momentum':.55 if r[2][0]>0 else .45,
                   'reversal':.45 if r[2][0]>0 else .55,
                   'logistic':sigmoid(w[0]+sum(w[j+1]*x[j] for j in range(4)))}
            for name,p in preds.items():totals[name]+=(p-y)**2
        folds.append({'start':dt.datetime.fromtimestamp(lo/1000,dt.timezone.utc).date().isoformat(),
                      'end':dt.datetime.fromtimestamp(hi/1000,dt.timezone.utc).date().isoformat(),
                      'trainExamples':len(train),'testExamples':len(test),
                      'brier':{name:round(val/len(test),6) for name,val in totals.items()}})
    if not folds:return {'status':'insufficient_data'}
    n=sum(f['testExamples'] for f in folds)
    return {'status':'evaluated','folds':folds,'testExamples':n,
            'aggregateBrier':{name:round(sum(f['brier'][name]*f['testExamples'] for f in folds)/n,6)
                              for name in ('constant','momentum','reversal','logistic')},
            'method':'Four expanding-window chronological folds, 2-day embargo, training-only scaling and L2 logistic regression',
            'caution':'Exploratory comparison, not untouched final validation. No demonstrated profitability.'}

def main():
    errors=[];raw={}
    for pair in PAIRS:
        candles={}
        for month in month_keys():
            try:
                for t,p in fetch(pair,month):candles[t//86400000]=(t,p)
            except Exception as exc:
                errors.append(f'{pair} {month}: {type(exc).__name__}')
        raw[pair]=sorted(candles.values())
        print(pair,len(raw[pair]),'days',flush=True)
    # Fail closed on missing/incomplete history. Never publish a misleading
    # successful evaluation after silent API errors or large data gaps.
    if errors:
        raise RuntimeError(f'Historical download incomplete: {len(errors)} failed monthly requests; first: {errors[:3]}')
    for pair, candles in raw.items():
        days=[t//86400000 for t,_ in candles]
        if len(days)<360:
            raise RuntimeError(f'{pair}: only {len(days)} daily candles (minimum 360)')
        gaps=[b-a for a,b in zip(days,days[1:])]
        if any(g!=1 for g in gaps):
            raise RuntimeError(f'{pair}: nonconsecutive daily candles detected')
        if len(set(days))!=len(days):
            raise RuntimeError(f'{pair}: duplicate candle dates')
    rows=[]
    for pair,candles in raw.items():
        prices=[p for _,p in candles]
        for i in range(14,len(candles)-1):
            if candles[i+1][0]-candles[i][0]>2*86400000:continue
            rows.append((candles[i][0],pair,features(prices,i),int(prices[i+1]>prices[i])))
    rows.sort(key=lambda x:x[0])
    report={'generatedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'insufficient_data',
            'source':'Binance Vision public historical daily spot candles, USDT-quoted, not GBP',
            'pairs':{k:len(v) for k,v in raw.items()},'examples':len(rows),
            'method':'Regularized logistic regression; past 1/3/7/14-day log returns',
            'validation':'Chronological 70/30 time split with 2-day embargo; one held-out period; no synthetic samples',
            'downloadErrors':errors[:30],
            'warning':'Research only. USDT prices are not GBP prices. A single holdout cannot establish predictive skill or profitable trading.'}
    if len(rows)>=600 and len({r[1] for r in rows})>=3:
        times=sorted(set(r[0] for r in rows))
        boundary=times[int(len(times)*.7)]
        train=[r for r in rows if r[0]<boundary-2*86400000]
        test=[r for r in rows if r[0]>=boundary]
        if len(train)>=300 and len(test)>=100:
            w=[0.]*5
            for _ in range(250):
                grad=[0.]*5
                for _,_,x,y in train:
                    pred=sigmoid(w[0]+sum(w[j+1]*x[j] for j in range(4)))
                    for j,z in enumerate([1.]+x):grad[j]+=(pred-y)*z
                for j in range(5):w[j]-=.35*(grad[j]/len(train)+(.01*w[j] if j else 0))
            base=sum(r[3] for r in train)/len(train)
            model=score(test,w)
            baseline=sum((base-r[3])**2 for r in test)/len(test)
            report.update(status='evaluated',trainExamples=len(train),testExamples=len(test),
                          testStart=dt.datetime.fromtimestamp(boundary/1000,dt.timezone.utc).date().isoformat(),
                          holdoutBrier=round(model,5),constantTrainRateBrier=round(baseline,5),
                          betterThanConstantBaseline=model<baseline)
    report['walkForward']=walk_forward(rows)
    if report['status']!='evaluated' or report['walkForward']['status']!='evaluated':
        raise RuntimeError('Insufficient validated history for a model evaluation')
    report['regimes']=regime_diagnostics(rows,raw)
    report['agentCouncil']=council_diagnostics(rows)
    report['trainedCouncil']=assess_trainable_council(rows)
    report['councilRobustness']=robustness(rows)
    report['councilEvaluation']=evaluate_council(rows,raw,report['walkForward'].get('folds',[]))
    report['selectiveForecasting']=selective_diagnostics(rows,report['walkForward'].get('folds',[]))
    report['researchVersion']='V31: specialist council robustness research'
    report['limitations']=[
      'Market-regime thresholds are hand-chosen, not learned or independently validated.',
      'Model candidates have been inspected on the same historical data; no untouched final evaluation exists.',
      'Daily candles cannot model intraday execution, spreads, fees, funding, or market impact.',
      'No real-money execution or automated live trading.'
    ]
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('Historical training:',report['status'],'examples:',len(rows),flush=True)
if __name__=='__main__':main()
