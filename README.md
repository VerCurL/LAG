# LAGLayer3

LAGLayer3 is a layered 4v4 air-combat control framework. The current first implementation contains the domain contracts, relative route representation, Provider seams, a configurable head/execution network template, a local simulation contract, the copied bsl_aggressive controller, and a short local smoke loop.

## Environment

Use the existing Conda environment:

```powershell
conda run -n jsbsim python ...
```

The environment currently provides Python 3.8, JSBSim 1.1.6, PyTorch, Gymnasium, NumPy and PyYAML.

## Smoke run

```powershell
$env:PYTHONPATH = "src"
conda run -n jsbsim python scripts/smoke_local.py --steps 120
conda run -n jsbsim python scripts/validate_config.py configs/default.yaml
```

The local adapter provides both a deterministic kinematic smoke backend and a JSBSim backend. The JSBSim backend uses the vendored data under third_party/jsbsim_data and currently drives the F16 model through the route-action contract.

## Source layout

The architecture and implementation order are documented in:

- docs/restructuring_decisions.md
- docs/directory_structure.md
- docs/implementation_handoff.md

The copied bsl_aggressive implementation lives under:

```text
src/laglayer3/flight_control/adapters/bsl_aggressive/
```

The public seam is BslAggressiveAdapter. Its copied weights and vendor files remain inside the new project.
