# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def move_model_to_device(maddpg, device):
    """Move all model modules to the same device (fix cpu/cuda mismatch)."""
    model_attributes = [
        "policy",
        "critic",
        "target_policy",
        "target_critic",
        "actor",
        "actor_target",
        "critic_target",
        "value_net",
        "log_std",
    ]

    def _move_attribute(obj, dev):
        """Move an object to a device if it has the 'to' method."""
        if obj is None:
            return
        if hasattr(obj, "to"):
            try:
                obj.to(dev)
            except Exception:
                pass

    # Move top-level model attributes
    for attr in model_attributes:
        _move_attribute(getattr(maddpg, attr, None), device)

    # Move agent attributes
    for agent in getattr(maddpg, "agents", []):
        for attr in model_attributes:
            _move_attribute(getattr(agent, attr, None), device)

    # Print device verification
    for agent in getattr(maddpg, "agents", []):
        for attr in ["policy", "actor", "critic"]:
            model = getattr(agent, attr, None)
            if model is not None and hasattr(model, "parameters"):
                try:
                    print(
                        f"First agent {attr} param device: {next(model.parameters()).device}")
                    return
                except StopIteration:
                    pass
