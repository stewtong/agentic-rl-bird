"""Official binary SQL EX metadata, using native async single/batch RM contracts."""

async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await reward_func(args, item, **kwargs) for item in sample]
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "reward" in metadata:
        value = float(metadata["reward"])
        if value not in (0.0, 1.0):
            raise ValueError("SQL EX reward must be binary")
        return value
    if metadata.get("ungradeable") == "max_turns":
        return 0.0
    raise RuntimeError("Missing explicit SQL domain grade")
