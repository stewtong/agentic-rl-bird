import asyncio,json
from types import SimpleNamespace
from miles.utils.types import Sample
from miles.rollout.rm_hub import async_rm,batched_async_rm
args=SimpleNamespace(custom_rm_path="bird_evaluator.rewards.reward_func")
def sample(metadata):
    return Sample(index=0,prompt="synthetic CPU fixture",metadata=metadata)
async def main():
    checks=[]
    for value in [0.0,1.0]:
        result=await async_rm(args,sample({"reward":value}))
        assert type(result) is float and result==value
        checks.append("single_"+str(value))
    group=[sample({"reward":0.0}),sample({"reward":1.0})]
    assert await batched_async_rm(args,group)==[0.0,1.0]
    checks.append("batch")
    assert await batched_async_rm(args,group,inplace_set_reward_field=True) is None
    assert [s.reward for s in group]==[0.0,1.0]
    checks.append("batch_inplace")
    assert await async_rm(args,sample({"ungradeable":"max_turns"}))==0.0
    checks.append("model_max_turns_zero")
    try: await async_rm(args,sample({}))
    except RuntimeError: checks.append("missing_grade_is_not_model_failure")
    else: raise AssertionError("Missing grade must not be silently scored")
    try: await async_rm(args,sample({"reward":0.5}))
    except ValueError: checks.append("nonbinary_rejected")
    else: raise AssertionError("Nonbinary grade must fail")
    print(json.dumps({"checks":checks,"passed":len(checks),"gpu_execution":False}))
asyncio.run(main())
