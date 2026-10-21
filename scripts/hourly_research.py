#!/usr/bin/env python3
"""Read-only scheduled forecast research. No orders, secrets, or exchange access."""
import json, pathlib, urllib.request, urllib.error, datetime, math, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'research-data.json'
now=datetime.datetime.now(datetime.timezone.utc)
ms=int(now.timestamp()*1000)
try: state=json.loads(PATH.read_text())
except (FileNotFoundError,ValueError): state={'observations':[],'forecasts':[]}
url='https://api.coingecko.com/api/v3/coins/markets?vs_currency=gbp&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h'
request=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'CryptoRadarResearch/0.1'})
coins=None
for attempt in range(4):
    try:
        with urllib.request.urlopen(request,timeout=30) as response:
            coins=json.load(response)
        break
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f'CoinGecko fetch attempt {attempt+1}/4 failed: {exc}', flush=True)
        if attempt == 3:
            raise
        # Short, bounded retry for transient network errors / API rate limits.
        time.sleep(5 * (attempt + 1))
if not isinstance(coins,list) or not coins: raise ValueError('No market data returned')
latest={}
for c in coins:
    if not isinstance(c,dict):continue
    p=c.get('current_price')
    if not isinstance(p,(int,float)) or not math.isfinite(p) or p<=0:continue
    latest[c['id']]=c
    state['observations'].append({'id':c['id'],'price':p,'time':ms})
history={}
for o in state['observations']:history.setdefault(o['id'],[]).append(o)
for cid,c in latest.items():
    older=[o for o in history[cid] if o['time']<=ms-86400000]
    if not older:continue
    if any(f['id']==cid and f['created']>ms-23*3600000 for f in state['forecasts']):continue
    # V16 research ensemble: independent, predeclared rules; NOT trained ML.
    # Every input is timestamped <= the forecast creation time.
    p=c['current_price']
    change=(p/older[-1]['price']-1)*100
    observations=history[cid]
    def past_at(hours):
        eligible=[o for o in observations if o['time']<=ms-hours*3600000]
        return eligible[-1]['price'] if eligible else None
    p48=past_at(48)
    # Distinct fixed research hypotheses; probabilities deliberately capped.
    momentum=max(.35,min(.65,.5+change*.012))
    mean_reversion=max(.35,min(.65,.5-change*.012))
    trend=max(.35,min(.65,.5+((p/p48-1)*100)*.006)) if p48 else .5
    # No selection or tuning using the existing evaluated holdout.
    models={'momentum':momentum,'mean_reversion':mean_reversion,'trend_48h':trend,'neutral':.5}
    ensemble=sum(models.values())/len(models)
    state['forecasts'].append({'id':cid,'symbol':c.get('symbol',''),'created':ms,'price':p,'pUp':ensemble,'models':models,'outcome':None,'scoredAt':None,'method':'V16 fixed untrained research ensemble'})
for f in state['forecasts']:
    if f.get('outcome') is not None or ms-f['created']<86400000:continue
    c=latest.get(f['id'])
    if c:
        f['outcome']=int(c['current_price']>f['price'])
        f['scoredAt']=ms
        f['scoredPrice']=c['current_price']
# Keep one observed price per UTC day and coin for 90 days to bound public Git history.
# Research training needs multiweek history; hourly collection still updates today's sample.
daily={}
for o in state['observations']:
    if o['time']>=ms-90*86400000:
        daily[(o['id'],o['time']//86400000)]=o
state['observations']=sorted(daily.values(),key=lambda o:(o['time'],o['id']))[-10000:]
state['forecasts']=state['forecasts'][-5000:]
state['lastRun']=now.isoformat()
PATH.write_text(json.dumps(state,separators=(',',':'))+'\n')
print('Recorded',len(latest),'coins;',len(state['forecasts']),'forecasts')
