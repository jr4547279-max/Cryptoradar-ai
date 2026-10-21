"""Chronological, training-only ensemble research; no live trading.

Each fold trains independent logistic specialists on past observations.
Weights are fixed equal weights (not fitted to holdout outcomes).
The last chronological 20% of days is reserved as an untouched-by-training
final test segment for this implementation; subsequent model selection using
that segment would invalidate its untouched status.
"""
import math, statistics
DAY=86400000

def sigmoid(z):
    return 1/(1+math.exp(-max(-25,min(25,z))))

SPECIALISTS={
    'short_term':(0,1),
    'medium_term':(1,2),
    'long_term':(2,3),
    'full_context':(0,1,2,3)
}

def fit(rows, indices, steps=160):
    mean=[statistics.fmean(r[2][j] for r in rows) for j in indices]
    std=[max(.00001,statistics.pstdev(r[2][j] for r in rows)) for j in indices]
    w=[0.]*(len(indices)+1)
    for _ in range(steps):
        gradient=[0.]*len(w)
        for r in rows:
            x=[(r[2][j]-mean[k])/std[k] for k,j in enumerate(indices)]
            p=sigmoid(w[0]+sum(a*b for a,b in zip(w[1:],x)))
            for k,z in enumerate([1.]+x):
                gradient[k]+=(p-r[3])*z
        for k in range(len(w)):
            w[k]-=.16*(gradient[k]/len(rows)+(.08*w[k] if k else 0))
    return mean,std,w

def predict(model, indices, row):
    mean,std,w=model
    x=[(row[2][j]-mean[k])/std[k] for k,j in enumerate(indices)]
    return sigmoid(w[0]+sum(a*b for a,b in zip(w[1:],x)))

def assess(rows):
    days=sorted({r[0] for r in rows})
    if len(days)<360:
        return {'status':'insufficient_data'}
    # All validation and final test predictions are made by models trained on
    # earlier dates only. No random split and no same-day label leakage.
    boundary=days[int(len(days)*.8)]
    train=[r for r in rows if r[0]<boundary-2*DAY]
    test=[r for r in rows if r[0]>=boundary]
    if len(train)<600 or len(test)<150:
        return {'status':'insufficient_data'}
    models={name:fit(train,cols) for name,cols in SPECIALISTS.items()}
    base=statistics.fmean(r[3] for r in train)
    squared={name:0. for name in ('baseline',*SPECIALISTS,'equal_weight_council')}
    loss={name:0. for name in squared}
    correct={name:0 for name in squared}
    bins=[{'n':0,'predictedSum':0.,'actualSum':0.} for _ in range(5)]
    for r in test:
        y=r[3]
        predictions={name:predict(models[name],cols,r) for name,cols in SPECIALISTS.items()}
        predictions['baseline']=base
        predictions['equal_weight_council']=sum(predictions[name] for name in SPECIALISTS)/len(SPECIALISTS)
        for name,p in predictions.items():
            squared[name]+=(p-y)**2
            loss[name]-=y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p))
            correct[name]+=int((p>=.5)==bool(y))
        p=predictions['equal_weight_council']
        b=min(4,int(p*5))
        bins[b]['n']+=1;bins[b]['predictedSum']+=p;bins[b]['actualSum']+=y
    n=len(test)
    scores={name:{'brier':round(squared[name]/n,6),
                  'logLoss':round(loss[name]/n,6),
                  'directionAccuracy':round(correct[name]/n,5)}
            for name in squared}
    return {
        'status':'evaluated',
        'trainExamples':len(train),'testExamples':n,
        'testStart':__import__('datetime').datetime.fromtimestamp(boundary/1000,__import__('datetime').timezone.utc).date().isoformat(),
        'agents':list(SPECIALISTS),
        'scores':scores,
        'councilBeatsBaselineBrier':scores['equal_weight_council']['brier']<scores['baseline']['brier'],
        'councilBeatsAllSpecialistsBrier':all(scores['equal_weight_council']['brier']<scores[name]['brier'] for name in SPECIALISTS),
        'calibration':[{'bucket':i,'count':b['n'],
                        'meanPrediction':round(b['predictedSum']/b['n'],5) if b['n'] else None,
                        'observedUpRate':round(b['actualSum']/b['n'],5) if b['n'] else None}
                       for i,b in enumerate(bins)],
        'method':'Four independent L2 logistic specialists trained on different past-return horizons; fixed equal-weight mean. Chronological final 20% test, two-day embargo, train-only standardization.',
        'warning':'This is a first exploratory final-period evaluation, not permanent untouched proof: once results are inspected, further changes require fresh future data for honest validation. No profit claim or live trading.'
    }
