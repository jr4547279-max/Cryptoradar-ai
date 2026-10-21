#!/usr/bin/env python3
"""Read-only scheduled forecast research. No orders, secrets, or exchange access."""
import json, pathlib, urllib.request, datetime, math
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'research-data.json'
now=datetime.datetime.now(datetime.timezone.utc)
ms=int(now.timestamp()*1000)
try: state=json.loads(PATH.read_text())
except (FileNotFoundError,ValueError): state={'observations':[],'forecasts':[]}
url='https://api.coingecko.com/api/v3/coins/markets?vs_currency=gbp&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h'
request=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'CryptoRadarResearch/0.1'})
with urllib.request.urlopen(request,timeout=30) as response: coins=json.load(response)
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
    momentum=(c['current_price']/older[-1]['price']-1)*100
    prob=max(.35,min(.65,.5+max(-.15,min(.15,momentum*.012))))
    state['forecasts'].append({'id':cid,'symbol':c.get('symbol',''),'created':ms,'price':c['current_price'],'pUp':prob,'outcome':None,'scoredAt':None})
for f in state['forecasts']:
    if f.get('outcome') is not None or ms-f['created']<86400000:continue
    c=latest.get(f['id'])
    if c:
        f['outcome']=int(c['current_price']>f['price'])
        f['scoredAt']=ms
        f['scoredPrice']=c['current_price']
state['observations']=[o for o in state['observations'] if o['time']>=ms-8*86400000][-20000:]
state['forecasts']=state['forecasts'][-5000:]
state['lastRun']=now.isoformat()
PATH.write_text(json.dumps(state,separators=(',',':'))+'\n')
print('Recorded',len(latest),'coins;',len(state['forecasts']),'forecasts')
