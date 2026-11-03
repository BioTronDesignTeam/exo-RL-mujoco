# Myo model experiments

## What is measured

| Item | Definition |
| --- | --- |
| Human | Reduced MyoLeg26 with 26 muscle/tendon actuators; recruitment controller follows a synthetic reference |
| Exo | Four hip/knee torque commands; MK1 mass approximately 19.1 kg |
| Primary effort | Mean squared leg-muscle activation, averaged over physics substeps; not metabolic energy or force |
| No-exo baseline | Exoskeleton removed; 25 kg backpack retained |
| Unpowered baseline | Exoskeleton worn with zero motor torque; backpack retained |
| BC / teacher | BC imitates simulation assistance; teacher uses privileged human-controller torque requests, not recorded human data |
| Comparison rule | Compare matched complete episodes; report tracking, ankle compensation, helper demand and motor demand alongside effort |

The simplified human uses a torque-based effort proxy. Its raw effort values cannot
be compared directly with Myo activation values. Muscle force, mechanical work and
joint moments are separate diagnostics; a reduction in activation is not an absolute
force reduction in newtons.

## Historical squat: assistance and muscle unloading

Positive percentages mean **less activation-squared effort**;
negative percentages mean more. Percentages below follow the saved summary's aggregation.

| Condition | Mean effort | Reduction vs no exo | Reduction vs unpowered exo | Motor torque RMS (Nm) |
| --- | ---: | ---: | ---: | ---: |
| No exo | 0.02655 | 0% | ? | 0 |
| Unpowered exo | 0.03769 | -41.9% | 0% | 0 |
| BC, 10 Nm cap | 0.03547 | -33.6% | +5.9% | 7.95 |
| Same BC, scaled to 40 Nm | 0.02437 | +8.2% | +35.3% | 31.84 |
| Recovered RL, 10 Nm, step 294k | 0.03694 | -39.1% | +2.0% | 0.21 |

The 40 Nm condition rescales a 10 Nm-trained BC policy. It is **not** a separately
trained 40 Nm policy. With the older reset/controller configuration, the historical
40 Nm RL checkpoint instead increased effort by 54.0% versus its own unpowered
baseline (39.96 Nm motor RMS).

| Muscle group | BC 10 Nm vs unpowered | Scaled BC 40 Nm vs unpowered | Scaled BC 40 Nm vs no exo |
| --- | ---: | ---: | ---: |
| Vasti | +40.7% | +83.1% | +69.7% |
| Rectus femoris | +44.7% | +93.7% | +87.8% |
| Adductors | +37.5% | +84.9% | +77.8% |
| Tibialis anterior + EDL | -10.8% | +2.6% | -26.6% |

Assistance did unload the intended thigh muscles. Ankle compensation and the cost
of wearing the exo can offset that benefit. Scaled BC also increased helper positive
work by 44.8% versus no exo, so the net activation improvement alone is insufficient
for a claim of physiological benefit. These are only two starts.

## Did RL improve on BC?

Initial residual pilot: two training seeds per algorithm, 1,200 residual decisions
per seed, four CPU environments, bounded +/-2 Nm corrections around **frozen BC**.
Evaluation used four fresh 24-second starts per policy. These results retain BC
underneath the learned correction; they do not demonstrate learning without BC.

| Controller | Activation reduction vs BC | Reduction vs no exo | BC-relative checks passed |
| --- | ---: | ---: | ---: |
| Frozen BC | 0% | +8.0% | 4/4 |
| SAC, seed 101 | +2.8% | +10.8% | 0/4 |
| SAC, seed 102 | +0.1% | +8.2% | 0/4 |
| TD3, seed 101 | -4.4% | +4.1% | 0/4 |
| TD3, seed 102 | -2.3% | +6.1% | 0/4 |

All completed, but completion alone did not satisfy the compensation/tracking/helper
screens. Schedule search found no eligible improvement and fell back to BC.

| Follow-up | Measured outcome |
| --- | --- |
| Plain residual replication | SAC: -0.2% to -3.5%; TD3: -2.5% to -6.5% activation reduction vs BC |
| Compensation-cost pilots | SAC: -0.7% to -2.7%; TD3: -2.9% to -11.0% vs BC; all passed 0/4 checks |
| Longer-credit BC-relative critic | Held-out pair ranking 56.6%; 0/12 guarded corrections accepted; actor stage blocked |
| Available local improvement | 7/12 validation states had a tested correction with positive screened gain; learning did not reliably identify them |

SAC showed a small benefit in one initial pilot, but no reliable, screened improvement
over BC. The evidence points to weak action guidance and compensation tradeoffs;
it does not establish that BC is optimal or that the metric alone caused RL degradation.
Follow-up evaluation used different fresh starts; compare methods within each study.


## Reproduce and interpret

```powershell
python scripts/view.py --config configs/myo/squat.yaml --task squat --mode off --seconds 24

python scripts/audit_policy.py --config configs/myolegs26_squat_stable.yaml --mode teacher --seeds 0 1 2 3 4 1000 1001 2000
```

- Myo physics runs on CPU; learner optimization can use CUDA. Current anatomy is
  planar, scaled geometrically by 1.157625, with native muscle force capacities.
- Current lumbar support is an ideal bounded helper (160 Nm cap), not an exo motor
  or muscle. Its work/torque must be included when assessing assistance.
- Use whole-model activation plus muscle-group unloading, tracking, compensation,
  helper demand and actuator demand. Current regression margins are experimental
  engineering checks, not physiological safety thresholds.

