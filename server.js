import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const PORT=Number(process.env.PORT||3000),DATA_DIR=process.env.DATA_DIR||'/tmp/cryptoradar';
const FILE=path.join(DATA_DIR,'research.json'),INTERVAL=Number(process.env.SCAN_INTERVAL_MS||3600000);
const HORIZON=86400000,MAX=5000;
let state={observations:[],forecasts:[],lastRun:null,lastError:null,version:1},busy=false;
try{const d=JSON.parse(fs.readFileSync(FILE,'utf8'));if(Array.isArray(d.observations)&&Array.isArray(d.forecasts))state={...state,...d}}catch{}
function persist(){try{fs.mkdirSync(DATA_DIR,{recursive:true});fs.writeFileSync(FILE+'.tmp',JSON.stringify(state));fs.renameSync(FILE+'.tmp',FILE)}catch(e){console.error('Storage unavailable:',e.message)}}
function probability(a,b){const change24=(a/b-1)*100;return Math.max(.35,Math.min(.65,.5+Math.max(-.15,Math.min(.15,change24*.012))))}
async function run(){if(busy)return;busy=true;try{
 const response=await fetch('https://api.coingecko.com/api/v3/coins/markets?vs_currency=gbp&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h',{headers:{accept:'application/json'},signal:AbortSignal.timeout(25000)});
 if(!response.ok)throw Error('Market API HTTP '+response.status);
 const data=await response.json();if(!Array.isArray(data)||!data.length)throw Error('Invalid market response');
 const now=Date.now(),latest=new Map();
 for(const q of data){if(typeof q.id!=='string'||!Number.isFinite(q.current_price)||q.current_price<=0)continue;latest.set(q.id,q);state.observations.push({id:q.id,price:q.current_price,time:now})}
 const byId=new Map();for(const o of state.observations){if(!byId.has(o.id))byId.set(o.id,[]);byId.get(o.id).push(o)}
 for(const q of latest.values()){const past=(byId.get(q.id)||[]).filter(o=>o.time<=now-HORIZON).at(-1);if(!past)continue;
 const exists=state.forecasts.some(f=>f.id===q.id&&f.created>now-23*3600000);if(!exists)state.forecasts.push({id:q.id,symbol:q.symbol,created:now,price:q.current_price,pUp:probability(q.current_price,past.price),outcome:null,scoredAt:null})}
 for(const f of state.forecasts){if(f.outcome!==null||now-f.created<HORIZON)continue;const q=latest.get(f.id);if(!q)continue;f.outcome=q.current_price>f.price?1:0;f.scoredAt=now;f.scoredPrice=q.current_price}
 state.observations=state.observations.filter(o=>o.time>=now-8*86400000).slice(-20000);
 state.forecasts=state.forecasts.slice(-MAX);state.lastRun=new Date(now).toISOString();state.lastError=null;persist();console.log('Scan',state.lastRun,'coins',latest.size,'forecasts',state.forecasts.length)
 }catch(e){state.lastError=String(e.message||e);console.error('Scan failed',state.lastError);persist()}finally{busy=false}}
function json(res,status,data){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','Access-Control-Allow-Origin':'*','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify(data))}
const server=http.createServer((req,res)=>{if(req.method!=='GET')return json(res,405,{error:'Read-only API'});const u=new URL(req.url,'http://localhost');if(u.pathname==='/health')return json(res,200,{ok:true,lastRun:state.lastRun,lastError:state.lastError,storage:DATA_DIR});
if(u.pathname==='/api/v1/summary'){const scored=state.forecasts.filter(f=>f.outcome===0||f.outcome===1),brier=scored.length?scored.reduce((sum,f)=>sum+(f.pUp-f.outcome)**2,0)/scored.length:null;return json(res,200,{updatedAt:state.lastRun,lastError:state.lastError,trackedCoins:new Set(state.observations.map(o=>o.id)).size,forecasts:state.forecasts.length,scored:scored.length,brier,baselineBrier:.25,method:'Experimental momentum baseline; not trained AI; no trading',warning:'Observation-based outcomes may be delayed; results not independent validation'})}
if(u.pathname==='/api/v1/forecasts'){const n=Math.max(1,Math.min(200,Number(u.searchParams.get('limit'))||50));return json(res,200,{forecasts:state.forecasts.slice(-n).reverse()})}
return json(res,404,{error:'Not found'})});
server.listen(PORT,'0.0.0.0',()=>console.log('CryptoRadar research backend listening on',PORT));
setInterval(run,Math.max(3600000,INTERVAL));run();
