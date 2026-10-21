"""Fail-closed autonomous learning status. Never promotes models or trades."""
import json, pathlib, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(name):
    try:return json.loads((ROOT/name).read_text())
    except (OSError,ValueError):return {}
training=read('training-report.json')
rewards=read('reward-report.json')
agents=rewards.get('agents') or {}
trained=training.get('status')=='evaluated'
beats=training.get('betterThanConstantBaseline') is True
reward_eligible=[name for name,result in agents.items() if result.get('eligible') is True]
status={
 'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'mode':'AUTONOMOUS_RESEARCH_ONLY',
 'dataCollection':'scheduled_hourly',
 'training':'evaluated' if trained else 'waiting_for_sufficient_forward_data',
 'historicalHoldoutBeatsBaseline':beats if trained else None,
 'rewardEligibleShadowAgents':reward_eligible,
 'automaticModelPromotion':False,
 'realTradingEnabled':False,
 'nextStep':'Keep collecting prospective evidence; independently validate new challengers before any promotion.',
 'caution':'Hourly evaluation is not continuous online learning. Shadow rewards do not update model weights.'
}
(ROOT/'learning-status.json').write_text(json.dumps(status,indent=2)+'\n')
print('Autonomous research:',status['training'],'reward eligible:',len(reward_eligible))
