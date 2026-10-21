"""Read-only integrity audit of the forward research record."""
import json, pathlib, math, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]

def audit(forecasts):
    problems=[]; seen=set(); matured=0; pending=0
    for i,f in enumerate(forecasts):
        if not isinstance(f,dict):
            problems.append({'index':i,'reason':'invalid_record'});continue
        key=(f.get('id'),f.get('created'))
        if key in seen:problems.append({'index':i,'reason':'duplicate_forecast'})
        seen.add(key)
        created=f.get('created'); scored=f.get('scoredAt'); outcome=f.get('outcome')
        price=f.get('price'); p=f.get('pUp'); models=f.get('models')
        if not isinstance(created,int) or isinstance(created,bool) or not isinstance(f.get('id'),str):
            problems.append({'index':i,'reason':'invalid_identity_or_time'})
        if not isinstance(price,(int,float)) or not math.isfinite(price) or price<=0:
            problems.append({'index':i,'reason':'invalid_price'})
        if not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
            problems.append({'index':i,'reason':'invalid_probability'})
        if isinstance(models,dict) and models:
            values=list(models.values())
            if not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in values):
                problems.append({'index':i,'reason':'invalid_specialist_probability'})
            elif isinstance(p,(int,float)) and math.isfinite(p) and abs(p-sum(values)/len(values))>1e-9:
                problems.append({'index':i,'reason':'council_probability_mismatch'})
        else:problems.append({'index':i,'reason':'missing_specialist_predictions'})
        if outcome is None:
            pending+=1
            if scored is not None:problems.append({'index':i,'reason':'unresolved_but_scored'})
        elif outcome in (0,1) and isinstance(scored,int) and isinstance(created,int) and scored-created>=86400000:
            matured+=1
            closing=f.get('scoredPrice')
            if not isinstance(closing,(int,float)) or not math.isfinite(closing) or closing<=0:
                problems.append({'index':i,'reason':'invalid_scored_price'})
            elif isinstance(price,(int,float)) and outcome!=int(closing>price):
                problems.append({'index':i,'reason':'outcome_price_mismatch'})
        else:problems.append({'index':i,'reason':'invalid_resolution'})
    return {'status':'PASS' if not problems else 'FAIL','forecasts':len(forecasts),
            'matured':matured,'pending':pending,'issues':problems[:100],
            'issueCount':len(problems),
            'limitation':'Checks internal consistency, not independent immutability or market data authenticity.'}

if __name__=='__main__':
    data=json.loads((ROOT/'research-data.json').read_text())
    result=audit(data.get('forecasts',[]))
    result['checkedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/'forecast-integrity-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Forecast integrity:',result['status'],result['issueCount'],'issues')
    if result['status']!='PASS':raise SystemExit(1)
