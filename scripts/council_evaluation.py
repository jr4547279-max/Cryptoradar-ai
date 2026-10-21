"""Out-of-sample council evaluation with fixed, auditable rules.

Research-only daily close-to-next-close returns. Each council decision uses
information available at the preceding close. This is NOT executable P&L.
"""
import math
from collections import defaultdict
from agent_council import council

DAY_MS=86400000

def evaluate_council(rows, raw, folds, roundtrip_cost=.004):
    if not 0 <= roundtrip_cost < .1:
        raise ValueError('Invalid cost assumption')
    lookup={}
    for pair,candles in raw.items():
        for i in range(len(candles)-1):
            t,p=candles[i];next_t,next_p=candles[i+1]
            if next_t-t==DAY_MS and p>0 and next_p>0:
                lookup[(pair,t)]=next_p/p-1
    summaries=[]
    for fold in folds:
        start=fold['start'];end=fold['end']
        from datetime import date
        lo=(date.fromisoformat(start)-date(1970,1,1)).days
        hi=(date.fromisoformat(end)-date(1970,1,1)).days
        events=[];blocked=defaultdict(int)
        for timestamp,pair,x,y in rows:
            day=timestamp//DAY_MS
            if not lo<=day<=hi:continue
            outcome=lookup.get((pair,timestamp))
            if outcome is None:continue
            decision=council(x,estimated_roundtrip_cost=roundtrip_cost)
            action=decision['decision']
            if action=='NO_TRADE':
                blocked[decision['reason']]+=1
            events.append((day,pair,action,outcome))
        if not events:continue
        selected=[(day,pair,action,ret) for day,pair,action,ret in events if action!='NO_TRADE']
        gross=sum((1 if action=='PAPER_LONG_CANDIDATE' else -1)*ret for day,pair,action,ret in selected)
        net=gross-roundtrip_cost*len(selected)
        # This is sum of hypothetical unit-notional returns per independent
        # signal, not a portfolio equity curve or a realizable return.
        correct=sum(((ret>0) if action=='PAPER_LONG_CANDIDATE' else (ret<0))
                    for _,_,action,ret in selected)
        summaries.append({
            'start':start,'end':end,'observations':len(events),
            'signals':len(selected),'coverage':round(len(selected)/len(events),5),
            'correctDirections':correct,
            'directionalHitRate':round(correct/len(selected),5) if selected else None,
            'sumGrossUnitNotionalReturns':round(gross,6),
            'sumNetUnitNotionalReturns':round(net,6),
            'assumedRoundTripCost':roundtrip_cost,
            'noTradeReasons':dict(blocked)
        })
    total_observations=sum(f['observations'] for f in summaries)
    total_signals=sum(f['signals'] for f in summaries)
    return {
        'status':'evaluated' if summaries else 'insufficient_data',
        'method':'Fixed council decisions evaluated on walk-forward test date windows; next-day close-to-close returns, fixed assumed round-trip cost.',
        'folds':summaries,
        'observations':total_observations,'signals':total_signals,
        'coverage':round(total_signals/total_observations,5) if total_observations else 0,
        'sumNetUnitNotionalReturns':round(sum(f['sumNetUnitNotionalReturns'] for f in summaries),6),
        'warning':'Exploratory and in-sample for the hand-chosen council rules, even though dates match walk-forward model test windows. Signals share days and correlated assets; summed unit-notional returns are NOT portfolio P&L. Close-to-close prices omit execution uncertainty, funding and borrow costs. No demonstrated profitability.'
    }
