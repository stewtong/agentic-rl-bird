import sys
if sys.version_info < (3, 11): raise SystemExit(f"Python 3.11 or newer is required (found {sys.version.split()[0]})")
from pathlib import Path
import subprocess,json,math
b=Path(__file__).resolve().parents[1]
o=b/'tests';cli=b/'scripts/analyze_bird.py';checks=[]
def run(name,mode='primary',bad=False):
 f=o/'fixtures'/('synthetic_'+name+'.jsonl')
 assert f.is_file(),f'missing fixture: {f}'
 args=[sys.executable,'-B',str(cli),mode,str(f)]
 if mode=='primary':
  rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
  seeds=sorted({x['training_seed'] for x in rows if x['training_seed'] is not None})
  if len(seeds)!=3:seeds=['seed-a','seed-b','seed-c']
  args+=['--seeds',*seeds]
 p=subprocess.run(args,text=True,capture_output=True)
 if bad:assert p.returncode!=0 and not p.stdout,(name,p.stdout,p.stderr)
 else:assert p.returncode==0,(name,p.stderr)
 checks.append({'case':name,'expected_rejection':bad,'exit_code':p.returncode,'passed':True})
 return p.stdout,json.loads(p.stdout) if p.returncode==0 else None
raw,agg=run('primary_aggregation');assert math.isclose(agg['pooled_question_analysis']['point_estimate'],1/6)
for name,gain in [('perfect_gain',1),('tied_gain',0),('reversed_gain',-1)]:
 _,v=run('primary_'+name);p=v['pooled_question_analysis'];assert p['point_estimate']==gain;assert p['interval_95_percentile']=={'low':gain,'high':gain};assert p['claim_rule']['passes']==(gain==1)
_,v=run('primary_known_mcnemar')
for seed in v['seed_specific'].values():assert seed['mcnemar']['exact_two_sided_p_value']==.5
_,v=run('sampling','sampling')
for c in v['conditions']:
 for seed in c['seeds']:
  for k,expected in [('1',5/24),('2',29/84),('4',52/105),('8',2/3)]:assert math.isclose(seed['oracle_pass_at_k'][k],expected)
raw2,_=run('primary_aggregation');assert raw2==raw
for name in ['duplicate','nonbinary','missing_seed','missing_pair','inconsistent_sample']:run('malformed_'+name,bad=True)
run('malformed_insufficient_attempts','sampling',bad=True)
print(f'Independent analysis verification: {len(checks)} CLI checks passed.')
