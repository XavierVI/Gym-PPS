



def train_ddpg_model(ddpg, config, buffers, agent_slices):
    """Perform training iterations on the DDPG model."""
    num_training_steps = config.n_updates_per_episode

    for _ in range(num_training_steps):
        ddpg.prep_training(device=DEVICE)

        # Train both controllers (predators and prey)
        for agent_idx in range(2):
            if len(buffers[agent_idx]) >= config.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = buffers[agent_idx].sample(
                    config.batch_size,
                    to_gpu=USE_CUDA
                )
                # Use compatibility layer for update
                _ddpg_update(ddpg, obs_batch, actions_batch, rewards_batch,
                             next_obs_batch, dones_batch, agent_idx, agent_slices)

        ddpg.update_all_targets()
        ddpg.prep_rollouts(device=DEVICE)



def _ddpg_update(ddpg, obs_s, acs_s, rews_s, next_obs_s, dones_s, agent_idx, agent_slices):
    """
    Call DDPG update() with compatibility layer.
    Supports different update signatures.
    """
    # Try obs_slices/act_slices style
    if hasattr(ddpg, "_obs_slices") and hasattr(ddpg, "_act_slices"):
        try:
            return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                               agent_i=agent_idx,
                               obs_slices=ddpg._obs_slices,
                               act_slices=ddpg._act_slices)
        except TypeError:
            pass

    # Try agent_slices style
    try:
        return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                           agent_i=agent_idx,
                           start_stop_num=agent_slices)
    except TypeError:
        pass

    # Plain update
    return ddpg.update(obs_s, acs_s, rews_s, next_obs_s, dones_s, agent_idx)


