# Reproducibility

1. Install the pinned native requirements in a Python 3.12 environment.
2. Obtain upstream models separately; match `asset_manifest.json` and retain their notices. The anatomy GLB is from MyoHub; additional surfaces and licenses are documented in the credits.
3. Rebuild local registration/initial-reference outputs using the model-foundation and controlled-jump scripts. A compiled MJB is not a portable model interchange format across arbitrary MuJoCo versions.
4. Resume a local experiment only after its `definition.json`, checkpoint and required initial references exist. This source package does not contain the original many-gigabyte rollout corpus or learned policy checkpoint. The supplied registered mechanical example can be analyzed independently of that history.
5. `configure_jump_three_metre_finish.py` applies the qualified 3m stopping rule. `compose_jump_rl_draft_v5.py` adds synchronized four-side and four-oblique comparison views after the approved opening.

The included metrics are retained evidence; they do not by themselves reproduce training. Recentring and population-based initialization are recorded separately from PPO gradient updates. A finite best candidate is not a global optimum.
