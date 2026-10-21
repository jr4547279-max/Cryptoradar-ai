"""Train-only specialist selection: nested chronological validation.

The winning specialist is selected using a past validation segment, never the
subsequent test segment. No optimization on test outcomes.
"""
from trainable_council import SPECIALISTS, fit, predict
from datetime import datetime,timezone
DAY=86400000

def assess_selection(rows):
    dates=sorted({r[0] for r in rows})
    if len(dates)<360:return {'status':'insufficient_data'}
    cutoffs=(.50,.65,.80)
    folds=[]
    for frac in cutoffs:
        i=int(len(dates)*frac)
        start=dates[i]
        stop=dates[min(len(dates)-1,i+max(1,int(len(dates)*.15))-1)]
        past=[r for r in rows if r[0]<start-2*DAY]
        test=[r for r in rows if start<=r[0]<=stop]
        past_days=sorted({r[0] for r in past})
        if len(past_days)<150 or len(test)<100:continue
        val_start=past_days[int(len(past_days)*.75)]
        fit_rows=[r for r in past if r[0]<val_start-2*DAY]
        validation=[r for r in past if r[0]>=val_start]
        if len(fit_rows)<200 or len(validation)<80:continue
        candidate_models={name:fit(fit_rows,cols) for name,cols in SPECIALISTS.items()}
        candidate_models['equal_weight_council']=None
        validation_scores={}
        for name in candidate_models:
            if name=='equal_weight_council':
                preds=[sum(predict(candidate_models[k],cols,r) for k,cols in SPECIALISTS.items())/len(SPECIALISTS) for r in validation]
            else:
                preds=[predict(candidate_models[name],SPECIALISTS[name],r) for r in validation]
            validation_scores[name]=sum((p-r[3])**2 for p,r in zip(preds,validation))/len(validation)
        winner=min(sorted(validation_scores),key=lambda name:validation_scores[name])
        # Refit chosen candidate on all pre-test history; do not inspect test.
        models={name:fit(past,cols) for name,cols in SPECIALISTS.items()}
        base=sum(r[3] for r in past)/len(past)
        sums={'baseline':0.,'selected':0.,'equal_weight_council':0.}
        for r in test:
            probs={name:predict(models[name],cols,r) for name,cols in SPECIALISTS.items()}
            ensemble=sum(probs.values())/len(probs)
            selected=ensemble if winner=='equal_weight_council' else probs[winner]
            for name,p in [('baseline',base),('selected',selected),('equal_weight_council',ensemble)]:
                sums[name]+=(p-r[3])**2
        folds.append({'start':datetime.fromtimestamp(start/1000,timezone.utc).date().isoformat(),
                      'selectedOnPastValidation':winner,
                      'validationExamples':len(validation),'testExamples':len(test),
                      'brier':{k:round(v/len(test),6) for k,v in sums.items()}})
    if len(folds)!=len(cutoffs):return {'status':'insufficient_data'}
    n=sum(f['testExamples'] for f in folds)
    aggregate={k:round(sum(f['brier'][k]*f['testExamples'] for f in folds)/n,6)
               for k in ('baseline','selected','equal_weight_council')}
    return {'status':'evaluated','folds':folds,'testExamples':n,'aggregateBrier':aggregate,
            'selectionBeatsBaseline':aggregate['selected']<aggregate['baseline'],
            'method':'Nested chronological validation: select among four specialists and equal-weight council using only earlier validation; refit on pre-test data; evaluate later period with two-day embargo.',
            'warning':'Exploratory historical research already inspected in prior versions; no untouched prospective validation, no profitability evidence.'}
