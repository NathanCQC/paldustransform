# paldustransform

A small, standalone [`pytket`](https://tket.quantinuum.com/api-docs/) implementation of the Paldus transform. It contains the transform and only the register-circuit helpers it needs

Requires Python 3.12 or newer and uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -c "from paldustransform import PaldusBox; print(PaldusBox(2).get_circuit())"
uv run pytest
```

To run the example notebook:

```bash
uv sync --extra examples
uv run jupyter lab examples/paldus_transform_example.ipynb
```

The public API is available directly from `paldustransform`:

```python
from paldustransform import NCounterBox, PaldusBox, PaldusStepBox
```
