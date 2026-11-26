# Storage policy

Generated model binaries use an immutable SHA-256 cache and per-case hard links. No shared model inode is overwritten. Changing storage does not change model bytes or dynamics.

`maintain_project_storage.py` first writes a reviewable cleanup plan. Referenced video cases, qualification reports, baseline inputs, policy checkpoints, settings, scores and histories are protected. `--apply` removes only unchanged generated files within the project results/deliverables trees. Final MP4s and named previews remain. Large intermediate model/motion arrays and obsolete frame caches are omitted from Git.

Future failed raw rollouts have a bounded per-worker retention window; selected video cases remain protected. This is not an archive of every raw historical attempt. Lightweight outcome/parameter records retain the optimization history.
