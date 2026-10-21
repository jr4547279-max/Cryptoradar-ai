"""Independent reward referee for locked paper forecasts. Never executes orders."""
import math

def referee(records, minimum=30):
    """Score every matured forecast; abstentions count toward coverage."""
    if not isinstance(records,list) or len(records)<minimum:
        return {'status':'INSUFFICIENT_DATA','reward':0,'eligible':False}
    seen=set()
    losses=[]; baseline=[]; decisions=0; abstentions=0
    for r in records:
        if not isinstance(r,dict):return {'status':'INVALID','reward':0,'eligible':False}
        key=(r.get('asset'),r.get('created'))
        if key in seen:return {'status':'INVALID_DUPLICATE','reward':0,'eligible':False}
        seen.add(key)
        p=r.get('probability'); y=r.get('outcome'); b=r.get('baseline',.5)
        created=r.get('created'); resolved=r.get('resolved')
        if (not isinstance(created,int) or not isinstance(resolved,int)
            or resolved-created<86400000 or y not in (0,1)
            or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in (p,b))):
            return {'status':'INVALID','reward':0,'eligible':False}
        losses.append((p-y)**2);baseline.append((b-y)**2)
        if r.get('abstain') is True:abstentions+=1
        elif r.get('abstain') is False:decisions+=1
        else:return {'status':'INVALID','reward':0,'eligible':False}
    n=len(records); coverage=decisions/n
    improvement=sum(baseline)/n-sum(losses)/n
    # Coverage is an eligibility requirement, never a reward to maximize.
    eligible=coverage>=.2 and improvement>0
    return {'status':'ELIGIBLE' if eligible else 'NO_REWARD',
            'eligible':eligible,'reward':round(min(1,max(0,improvement*10)),6) if eligible else 0,
            'examples':n,'coverage':round(coverage,4),
            'brier':round(sum(losses)/n,6),'baselineBrier':round(sum(baseline)/n,6),
            'note':'Experimental reward score only; no training or live trading permission.'}
