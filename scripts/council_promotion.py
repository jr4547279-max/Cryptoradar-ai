"""Conservative research promotion gate. No live trading activation."""
def assess_promotion(report):
    trained=report.get('trainedCouncil') or {}
    robust=report.get('councilRobustness') or {}
    scores=trained.get('scores') or {}
    base=scores.get('baseline',{}).get('brier')
    council=scores.get('equal_weight_council',{}).get('brier')
    windows=robust.get('folds') or []
    historical_ok=(trained.get('status')=='evaluated' and
                   robust.get('status')=='evaluated' and
                   isinstance(base,(float,int)) and isinstance(council,(float,int)) and
                   council<base and len(windows)>=3 and
                   all(f['brier']['council']<f['brier']['baseline'] for f in windows))
    reasons=[]
    if not historical_ok:
        reasons.append('Council has not consistently beaten the baseline across required historical evaluations.')
    reasons.append('No independently verified, immutable forward-forecast performance record for this council.')
    reasons.append('No independently validated, executable net-profit and drawdown evidence.')
    return {
        'status':'RESEARCH_ONLY',
        'paperTradePromotionApproved':False,
        'realMoneyApproved':False,
        'historicalChecksPassed':historical_ok,
        'reasons':reasons,
        'rule':'Historical backtests alone can never authorize a promotion; independent forward testing and risk review are required.'
    }
