"""Editor startup: wait a few seconds for the editor to settle, then build the LiveDirt level and follow the bridge."""
import unreal

_n = {"frames": 0, "handle": None}


def _later(dt):
    _n["frames"] += 1
    if _n["frames"] < 180:
        return
    unreal.unregister_slate_post_tick_callback(_n["handle"])
    try:
        import livedirt_ue
        livedirt_ue.start()
    except Exception as e:
        unreal.log_warning(f"[LiveDirt] auto-start failed ({e}). Start the bridge (unreal/run_bridge.sh), then in the Python console: import livedirt_ue; livedirt_ue.start()")


_n["handle"] = unreal.register_slate_post_tick_callback(_later)
