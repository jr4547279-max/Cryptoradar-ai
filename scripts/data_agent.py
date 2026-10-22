#!/usr/bin/env python3
"""Read-only Data Agent: checks research evidence structure without altering it."""
import datetime, json, math, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
DAY=86_400_000

def audit_data(data, now_ms=None):
    now_ms = now_ms if now_ms is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp()*1000)
    errors, warnings, seen, daily, coins, days = [], [], set(), set(), set(), set()
    observations = data.get("observations", []) if isinstance(data,dict) else []
    forecasts = data.get("forecasts", []) if isinstance(data,dict) else []
    if not isinstance(data,dict):
        return {"status":"FAIL","errors":["root_not_object"],"warnings":[],"observations":0,"validObservations":0,"coins":0,"utcDays":0,"forecasts":0,"maturedForecasts":0,"pendingForecasts":0}
    if not isinstance(observations,list): errors.append("observations_not_list"); observations=[]
    if not isinstance(forecasts,list): errors.append("forecasts_not_list"); forecasts=[]
    valid=0; latest=None
    for i,row in enumerate(observations):
        if not isinstance(row,dict): errors.append(f"observation_{i}_not_object"); continue
        cid,p,t=row.get("id"),row.get("price"),row.get("time")
        if not isinstance(cid,str) or not cid.strip(): errors.append(f"observation_{i}_invalid_coin_id"); continue
        if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or p<=0: errors.append(f"observation_{i}_invalid_price"); continue
        if isinstance(t,bool) or not isinstance(t,int) or t<=0: errors.append(f"observation_{i}_invalid_timestamp"); continue
        if t>now_ms+300_000: errors.append(f"observation_{i}_timestamp_in_future"); continue
        key=(cid,t)
        if key in seen: errors.append(f"observation_{i}_duplicate_coin_timestamp"); continue
        seen.add(key); day=(cid,t//DAY)
        if day in daily: warnings.append(f"observation_{i}_multiple_samples_same_coin_utc_day")
        daily.add(day); valid+=1; coins.add(cid); days.add(t//DAY)
        latest=t if latest is None else max(latest,t)
    matured=pending=0; fseen=set()
    for i,row in enumerate(forecasts):
        if not isinstance(row,dict): errors.append(f"forecast_{i}_not_object"); continue
        cid,created,p,prob=row.get("id"),row.get("created"),row.get("price"),row.get("pUp")
        if not isinstance(cid,str) or not cid.strip() or isinstance(created,bool) or not isinstance(created,int) or created<=0:
            errors.append(f"forecast_{i}_invalid_identity_or_timestamp"); continue
        key=(cid,created)
        if key in fseen: errors.append(f"forecast_{i}_duplicate_coin_timestamp")
        fseen.add(key)
        if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or p<=0: errors.append(f"forecast_{i}_invalid_price")
        if isinstance(prob,bool) or not isinstance(prob,(int,float)) or not math.isfinite(prob) or not 0<=prob<=1: errors.append(f"forecast_{i}_invalid_probability")
        outcome,scored=row.get("outcome"),row.get("scoredAt")
        if outcome is None:
            pending+=1
            if scored is not None: errors.append(f"forecast_{i}_pending_but_has_scored_at")
        elif outcome in (0,1) and not isinstance(outcome,bool) and isinstance(scored,int) and scored>=created+DAY:
            close=row.get("scoredPrice")
            if isinstance(close,bool) or not isinstance(close,(int,float)) or not math.isfinite(close) or close<=0: errors.append(f"forecast_{i}_invalid_scored_price")
            elif isinstance(p,(int,float)) and outcome!=int(close>p): errors.append(f"forecast_{i}_outcome_disagrees_with_prices")
            else: matured+=1
        else: errors.append(f"forecast_{i}_invalid_resolution")
    if valid==0: warnings.append("no_valid_observations")
    if len(days)<21: warnings.append("fewer_than_21_distinct_utc_days")
    if len(coins)<10: warnings.append("fewer_than_10_distinct_coins")
    if not forecasts: warnings.append("no_forecasts_yet")
    if latest is not None and now_ms-latest>36*60*60*1000: warnings.append("latest_observation_older_than_36_hours")
    return {"status":"PASS" if not errors else "FAIL","checkedAt":datetime.datetime.fromtimestamp(now_ms/1000,datetime.timezone.utc).isoformat(),
      "sourceDeclaredByCollector":"CoinGecko public markets endpoint; source authenticity is not independently verified",
      "observations":len(observations),"validObservations":valid,"coins":len(coins),"utcDays":len(days),"latestObservationMs":latest,
      "forecasts":len(forecasts),"maturedForecasts":matured,"pendingForecasts":pending,
      "errorCount":len(errors),"errors":errors[:100],"warningCount":len(warnings),"warnings":warnings[:100],
      "limitations":["PASS checks structural consistency only, not external-source authenticity.","Sparse history is warned about, never filled with synthetic data.","This agent does not train models, edit observations, score strategy performance, or place orders."]}

def main():
    data=json.loads((ROOT/"research-data.json").read_text(encoding="utf-8"))
    report=audit_data(data)
    (ROOT/"data-agent-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Data Agent: {report['status']} | {report['validObservations']}/{report['observations']} valid observations | {report['coins']} coins | {report['utcDays']} UTC days | {report['errorCount']} errors | {report['warningCount']} warnings")
    if report["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
