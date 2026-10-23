# Recorded environment

`requirements.lock.txt` is copied byte-for-byte from the author-provided environment export. It is an observed package inventory, not an independently rebuilt or minimal dependency set. `environment-record.json` retains the recorded version/hardware checks with the private executable path removed. `formal-v2-runtime.json` records the actual formal-v2 runner environment: Python 3.12.13, PyTorch 2.2.2, NumPy 1.26.4, CPU, one intra-op and one inter-op thread.

The offline numerical verification commands use Python 3 standard library only and do not require this environment. No dependency installation, source-model import, checkpoint deserialization, model training or inference was performed when assembling this version.
