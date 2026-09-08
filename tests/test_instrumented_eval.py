import sys
import types


# The local unit-test environment intentionally avoids the heavyweight torch
# dependency; this test exercises only the output-filtering context manager.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.is_tensor = lambda value: False
    sys.modules["torch"] = torch_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    class _Tqdm:
        @classmethod
        def write(cls, message, **kwargs):
            return None

    tqdm_stub.tqdm = _Tqdm
    sys.modules["tqdm"] = tqdm_stub

from cl_sam_replication.instrumented_eval import compact_router_output


def test_compact_router_output_suppresses_batch_messages(monkeypatch):
    from tqdm import tqdm

    messages = []
    monkeypatch.setattr(tqdm, "write", lambda message, **kwargs: messages.append(message))

    with compact_router_output():
        tqdm.write("[VAE-LOGITS][batch 0] noisy")
        tqdm.write("[VAE-LOGITS][global] retained")

    assert messages == ["[VAE-LOGITS][global] retained"]


def test_verbose_router_output_is_unchanged(monkeypatch):
    from tqdm import tqdm

    messages = []
    monkeypatch.setattr(tqdm, "write", lambda message, **kwargs: messages.append(message))

    with compact_router_output(enabled=False):
        tqdm.write("[VAE-LOGITS][batch 0] retained")

    assert messages == ["[VAE-LOGITS][batch 0] retained"]
