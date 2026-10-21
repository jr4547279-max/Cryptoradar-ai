"""CryptoRadar experimental multi-agent council.

Deterministic, auditable specialist opinions from past-return features.
Not autonomous LLM agents, not a trained ensemble, not live trading.
"""
import math

def council(features, data_valid=True, estimated_roundtrip_cost=0.004):
    if not data_valid or len(features)!=4 or any(not math.isfinite(v) for v in features):
        return {'decision':'NO_TRADE','reason':'Missing or invalid market data','agents':{}}
    r1,r3,r7,r14=features
    # Each specialist uses a fixed heuristic; these are not calibrated probabilities.
    agents={
        'trend':{'vote':'UP' if r7>0.02 and r14>0.03 else 'DOWN' if r7<-.02 and r14<-.03 else 'WAIT',
                 'evidence':'7-day and 14-day trailing returns'},
        'reversal':{'vote':'DOWN' if r1>.04 else 'UP' if r1<-.04 else 'WAIT',
                    'evidence':'1-day move beyond ±4%'},
        'regime':{'vote':'CAUTION' if abs(r1)>.06 or abs(r7)>.16 else 'NORMAL',
                  'evidence':'large recent absolute price moves'},
        'risk':{'vote':'BLOCK' if abs(r1)>.09 or abs(r7)>.22 else 'PASS',
                'evidence':'conservative shock thresholds'},
        'execution':{'vote':'BLOCK' if estimated_roundtrip_cost>=.01 else 'PASS',
                     'evidence':'assumed total round-trip trading costs'},
    }
    directional=[agents['trend']['vote'],agents['reversal']['vote']]
    if agents['risk']['vote']=='BLOCK' or agents['execution']['vote']=='BLOCK':
        decision='NO_TRADE';reason='Risk or estimated execution cost gate blocked'
    elif agents['regime']['vote']=='CAUTION':
        decision='NO_TRADE';reason='Volatile market regime: abstain'
    elif directional.count('UP')==2:
        decision='PAPER_LONG_CANDIDATE';reason='Two directional specialists agree; research only'
    elif directional.count('DOWN')==2:
        decision='PAPER_SHORT_CANDIDATE';reason='Two directional specialists agree; research only'
    else:
        decision='NO_TRADE';reason='Specialists disagree or have insufficient conviction'
    # Fixed trend and reversal rules usually conflict or abstain: expose this honestly.
    return {'decision':decision,'reason':reason,'agents':agents,
            'status':'EXPERIMENTAL_UNVALIDATED',
            'warning':'Not a calibrated prediction, not a trade instruction.'}

def council_diagnostics(rows):
    counts={}
    for _,_,x,_ in rows:
        d=council(x)['decision']
        counts[d]=counts.get(d,0)+1
    return {'status':'experimental','sampleCount':len(rows),'decisions':counts,
            'description':'Five fixed-rule specialists; disagreement and risk gates default to no trade.',
            'limitations':'Not trained, not validated as profitable, and does not place orders.'}
