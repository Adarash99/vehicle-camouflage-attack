# camouflage

Research project generating adversarial textures that fool vehicle object detectors while looking like natural patterns (snow/dirt). Uses CARLA simulator + neural renderer + adversarial optimization.

## Status

**Phase 1 (Foundation): 75% Complete**

✅ Completed:
- Neural renderer trained (MSE: 0.000296) and differentiability verified
- PyTorch EfficientDet-D0 integrated with pre-NMS access
- Attack loss function implemented
- EOT training loop with 6-viewpoint sampling
- Camera viewpoint control in CarlaHandler

⏳ Next Steps:
- Run first optimization with random textures
- Hyperparameter tuning
- Evaluation metrics implementation

## Quick Start

```bash
# Start CARLA server
./CarlaUE4.sh

# Run Phase 1 training
conda activate camo
python experiments/phase1_random.py
```

See `CLAUDE.md` for full documentation and `docs/plans/2026-01-30-adversarial-camouflage-design.md` for detailed plan.