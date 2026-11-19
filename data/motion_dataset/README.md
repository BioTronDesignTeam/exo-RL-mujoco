# Converted motion data collection

This archive contains exported trajectories, not trained controllers or validated Myo motions.

- gait120/: all 6,882 published JointAngle MOT segments, 120 subjects, seven activities; full-body native signs, degrees, seconds. CC BY 4.0; cite https://doi.org/10.6084/m9.figshare.27677016.v1 and https://doi.org/10.1038/s41597-025-05391-0.
- official_examples/: native CSV conversions of supplied MOT/STO tables from three pinned official repositories. Includes coordinate states, forces, controls, duplicate simulation outputs and nonhuman/simplified tutorial models; consult manifest table_kind and units before use.
- official_degrees/: explicitly unit-labelled anatomical angular positions exported in degrees, with native signs retained. Translations/speeds/forces are not joint angles.
- opensim_gait2392/, opensim_hamner/, opencap_squat/: curated walking, running and full-body squat exports with sidecar metadata.
- gatech_2023/, candidates/, cycles/: previously collected Georgia Tech trials, approved labelled quick-step approximations and selected measured intervals. CC BY 4.0; cite https://doi.org/10.35090/gatech/70296.
- sources/: original motion files, available licensing/provenance, Gait120 demographics/anthropometrics and processing code. EMG and marker archives were not downloaded wholesale.

Gait120 and curated exports have physical timestamps, not invented gait-phase durations. Times are zeroed per file; original start times are retained in metadata. Gait120 step segments need not be full strides. Additional native coordinates are not simulator torso/root targets until retargeted. Source/model knee signs differ. Official example tables with repeated timestamps are preserved and flagged; do not feed them directly to interpolation without explicit handling. Quaternion components are not joint flexion angles.

Repository software licenses do not establish a blanket license for all example recordings. OpenSim-models has no repository-wide license returned by its license endpoint; consult original file/model credits and establish redistribution permission for your use. Apache-2.0 texts for OpenSim Core/OpenCap code are included. No third-party license is replaced by this bundle.

Scope: all Gait120 JointAngle MOT files in 12 published archives; all MOT/STO files in pinned opensim-models/opencap-processing and published examples in opensim-core. Core regression-test fixtures are inventoried but excluded. Empty storage files cannot yield trajectories. This is not a claim to include every OpenSim-compatible dataset on the internet. Models or solver examples without supplied trajectories were not run to manufacture recordings.

See recorded/catalog.json and dataset manifests for source URLs, commits, file hashes, units, processing and exclusions. SHA256SUMS.json covers every included recorded file. No new training or dynamics simulation was run for this export.
