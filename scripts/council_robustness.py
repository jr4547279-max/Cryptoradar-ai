"""Diagnostic council robustness checks, not model selection or live trading."""
import math
from trainable_council import SPECIALISTS, fit, predict

DAY=86400000

def robustness(rows):
    dates=sorted({r[0] for r in rows})
    if len(dates)<360:return {'status':'insufficient_data'}
    # Predefined chronological evaluation windows, each model fit only on earlier dates.
    cutoffs=(.50,.65,.80)
    folds=[]
    for frac in cutoffs:
        i=int(len(dates)*frac)
        start=dates[i]
        stop=dates[min(len(dates)-1,i+max(1,int(len(dates)*.15))-1)]
        train=[r for r in rows if r[0]<start-2*DAY]
        test=[r for r in rows if start<=r[0]<=stop]
        if len(train)<400 or len(test)<100:continue
        models={name:fit(train,cols) for name,cols in SPECIALISTS.items()}
        base=sum(r[3] for r in train)/len(train)
        brier={name:0. for name in ('baseline','council',*SPECIALISTS)}
        by_pair={}
        for r in test:
            y=r[3];pair=r[1]
            probs={name:predict(models[name],cols,r) for name,cols in SPECIALISTS.items()}
            probs['baseline']=base
            probs['council']=sum(probs[name] for name in SPECIALISTS)/len(SPECIALISTS)
            pair_stats=by_pair.setdefault(pair,{'n':0,'councilSquaredError':0.,'baselineSquaredError':0.})
            pair_stats['n']+=1
            pair_stats['councilSquaredError']+=(probs['council']-y)**2
            pair_stats['baselineSquaredError']+=(base-y)**2
            for name,p in probs.items():brier[name]+=(p-y)**2
        folds.append({
            'start':__import__('datetime').datetime.fromtimestamp(start/1000,__import__('datetime').timezone.utc).date().isoformat(),
            'trainExamples':len(train),'testExamples':len(test),
            'brier':{k:round(v/len(test),6) for k,v in brier.items()},
            'perAsset':{pair:{'examples':v['n'],'councilBrier':round(v['councilSquaredError']/v['n'],6),
                               'baselineBrier':round(v['baselineSquaredError']/v['n'],6)}
                        for pair,v in by_pair.items()}
        })
    if len(folds)!=len(cutoffs):return {'status':'insufficient_data'}
    n=sum(f['testExamples'] for f in folds)
    agg={name:round(sum(f['brier'][name]*f['testExamples'] for f in folds)/n,6)
         for name in ('baseline','council',*SPECIALISTS)}
    return {
        'status':'evaluated','folds':folds,'testExamples':n,
        'aggregateBrier':agg,
        'councilBeatsBaseline':agg['council']<agg['baseline'],
        'foldsBeatingBaseline':sum(f['brier']['council']<f['brier']['baseline'] for f in folds),
        'method':'Three expanding chronological windows with two-day embargo and per-asset diagnostics; each model fitted on past data only.',
        'warning':'Exploratory windows on already inspected historical data. Overlapping research with previous evaluations means these are NOT new independent confirmations.'
    }
