"""Deterministic unit tests for Data Agent; fixtures are never model training data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from data_agent import audit_data
NOW=1_800_000_000_000
DAY=86_400_000

def test():
    clean={"observations":[{"id":"bitcoin","price":100+i,"time":NOW-i*DAY} for i in range(3)],
           "forecasts":[{"id":"bitcoin","created":NOW-DAY,"price":100,"pUp":0.55,"outcome":None,"scoredAt":None}]}
    r=audit_data(clean,NOW)
    assert r["status"]=="PASS" and r["validObservations"]==3 and r["pendingForecasts"]==1
    assert "fewer_than_21_distinct_utc_days" in r["warnings"]
    r=audit_data({"observations":[{"id":"bitcoin","price":float("nan"),"time":NOW},{"id":"ethereum","price":10,"time":NOW+400_000}],"forecasts":[]},NOW)
    assert r["status"]=="FAIL" and r["errorCount"]==2
    row={"id":"bitcoin","price":100,"time":NOW-DAY}
    r=audit_data({"observations":[row,dict(row)],"forecasts":[]},NOW)
    assert r["status"]=="FAIL" and any("duplicate_coin_timestamp" in e for e in r["errors"])
    f={"id":"bitcoin","created":NOW-2*DAY,"price":100,"pUp":0.7,"outcome":1,"scoredAt":NOW,"scoredPrice":90}
    r=audit_data({"observations":[],"forecasts":[f]},NOW)
    assert r["status"]=="FAIL" and "forecast_0_outcome_disagrees_with_prices" in r["errors"]
    f={"id":"bitcoin","created":NOW-DAY,"price":100,"pUp":1.5,"outcome":None,"scoredAt":None}
    r=audit_data({"observations":[],"forecasts":[f]},NOW)
    assert r["status"]=="FAIL" and "forecast_0_invalid_probability" in r["errors"]
    print("Data Agent tests passed (5 cases)")
if __name__=="__main__": test()
