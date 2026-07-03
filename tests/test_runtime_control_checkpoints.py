from capx.runtime_control.checkpoints import NamespaceCheckpointStore


def test_checkpoint_restores_copyable_globals():
    store = NamespaceCheckpointStore()
    namespace = {"x": 1, "__name__": "__main__"}
    checkpoint_id = store.save("before_region_2", namespace)
    namespace["x"] = 99

    restored = store.restore(checkpoint_id)

    assert restored["x"] == 1
