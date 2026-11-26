# Method and interpretation

The standing-long-jump study uses native MuJoCo forward dynamics: three unactuated sagittal root coordinates, eleven bounded internal joint torques, gravity, friction and rigid-foot contacts. The takeoff impulse, ground reactions and airborne centre-of-mass trajectory emerge from the simulation. Air drag and toe articulation are omitted.

A 9→64→64→58 PPO parameter policy selects a complete posture/timing/drive sequence. A PD/state-feedback controller executes it. Supplemental numerical search adjusted landing posture under the same torque capacities. The adopted result did not increase feedback gains, alter contact settings or relax eligibility thresholds.

Qualification requires retained distance ≥3.000 m and stable recovery at 0.5, 0.25, 0.125 and 0.0625 ms. All four were repeated from rest without launch caching. The minimum was 3.008973848 m. Numerical screens include ROM, conservation and a fixed heuristic pre-takeoff GRF screen below 5.5 bodyweights. That screen is not a universal human capacity law. This result does not establish global optimality.

At qualification, 9,247 V3 PPO rollouts and 80 distinct supplemental parameter proposals had completed. Final checkpoint/worker shutdown ended at 9,258 PPO rollouts, 192 policy updates and 84 supplemental proposals / 339 supplemental native simulations. Earlier inherited training is separately recorded in the source branch. Simulations repeated at different time steps are not new PPO updates.

Muscle tensions are inverse postprocessing over 438 model paths, using torque/support equilibrium, passive floors and regularized force sharing. Individual-muscle force ceilings are not enforced. These forces do not drive the forward learner, and do not represent measured activation or minimum physiological effort. The forward net joint torque limits and inverse muscle-force allocation are separate assumptions.

The older jump/squat and box-jump comparisons use reference-motion/inverse-dynamics pipelines with their own disclosed approximations. Their absolute loads should not be treated as independently validated human measurements or directly pooled with the forward-jump study.

The final long-jump video is 59.8 s. Its approved first33s are preserved; the following presentation is faster by1.28955×. Simulation time, force amplitudes and all numerical outcomes are unchanged. Force-time integrals refer to physical simulation time, not playback duration or replay count. Joint reaction resultants are forces, not contact pressure.
