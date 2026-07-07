def input_node(state, config):
    return {"idx": state.get("idx", state.get("metadata", {}).get("idx", -1))}
