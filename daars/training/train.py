# this is a script to run the training with daars rewards

def train_agent(
    config: dict,
    reward_type: str
) -> dict:
    # let's train one agent first

    tcfg     = config["training"]
    total_ts = int(tcfg["total_timesteps"])

    is_sac   = reward_type == "sac_lag"
    is_lag   = reward_type in ("ppo_lag", "sac_lag", "daars_lag")

    # if not sac single env.
    if is_sac:
        n_envs   = 1

    print(f"\n>> {reward_type:12s}  seed={seed}  "
          f"envs={n_envs}  steps={total_ts:,}  ({scenario})")


    return {"agent_data": "data"}