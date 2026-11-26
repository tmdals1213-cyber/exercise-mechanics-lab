# Anatomical surface assets

Muscle geometry and names: **BodyParts3D © The Database Center for Life Science (DBCLS), CC BY 4.0**. Mitsuhashi N, et al. BodyParts3D: 3D structure database for anatomical concepts. Nucleic Acids Res. 2009;37:D782–5. DOI: 10.1093/nar/gkn613.

Obtained through human-atlas and MyoHub/myosuite-assets. The MyoHub asset is posed, skinned and chained onto the MyoSuite whole-body skeleton. The body skin is adapted from MakeHuman/MPFB, CC0 1.0. These combined files are distributed by MyoHub under CC BY 4.0.

Source: https://huggingface.co/datasets/myohub/myosuite-assets

File: composite/fullbody_muscles_skin.glb

SHA-256: 27d77489d748cbdc8c6eb5ad9ae28ee7cf705d2473777f1430dadc31b85ee303

Our changes: retained actual muscle surfaces, masked the skin to parts of the face/hands/feet, exposed the posterior neck/trapezius, drove the skeleton with MyoSim inverse-kinematics poses, assigned calculated tension colours to explicitly matched muscle groups, and rendered a portrait OHP. Unmatched anatomy is grey-brown. The meshes are not themselves a constitutive tissue-mechanics model.

Presentation revision: enlarged selected muscle belly cross-sections with a smooth taper to the long-axis ends, without modifying the original GLB, skeleton, joint centres or force solver. The left/right platysma surfaces are hidden to expose the clavicular pectoralis layer. Per-mesh changes are recorded in results/ohp_fullbody/visual_anatomy.json. Visual enlargement does not represent a measured increase in muscle mass or strength.

## Supplementary components — 2026-11-17

Source: [Johan Bellander / BodyExplorer](https://github.com/JohanBellander/BodyExplorer/tree/7d04bf3c4de2bd9cb234dd51d7e6857c099afafd), public/anatomy.glb and public/mesh_mapping.json, pinned commit 7d04bf3c4de2bd9cb234dd51d7e6857c099afafd. The extracted surface data and source hash are in torso_supplement.json.

- 18 Z-Anatomy components: bilateral latissimus dorsi, rectus abdominis, internal oblique, transversus abdominis, quadratus lumborum, multifidus cervicis/thoracis/lumborum, rotatores. **Z-Anatomy, The Libre 3D Atlas**, Gauthier Kervyn, Marcin Zielinski, Lluis Vinent; underlying BodyParts3D credits Kousaku Okubo and DBCLS. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Upstream licence](https://github.com/Z-Anatomy/Models-of-human-anatomy/blob/master/License.txt).
- 2 BP3D components: bilateral tensor fasciae latae. **BodyParts3D © DBCLS**, [CC BY-SA 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en), as identified by the BodyExplorer component source mapping. These are distinct from the base MyoHub asset's stated distribution licence.

Changes: axis-preserving torso registration, presentation-only cross-section enlargement, trunk-restricted skeletal weights, and matching the narrow latissimus tendon end to the native MyoSim proximal-humerus LAT1/2/3 insertion sites. No distal-arm attachment. Joint/pose checking is not a validation of full soft-tissue deformation. Preserve the individual component notices and applicable licences when sharing derivatives. The original base GLB remains unchanged.

Additional display registration: the lower rectus abdominis cross-sections are shifted to match the existing anterior abdominal wall envelope while retaining the source surface and tendinous inscriptions. No muscle-force or strength change.


Measured CMJ: Calisti, Mohr & Federolf (2025), https://doi.org/10.6084/m9.figshare.28890545, CC BY 4.0.
COM power analysis and natural-speed composition are derived outputs.

## Calculation model and reference cards — 2026-11-20

- Adapted **MyoSim `myofullbody`**, MyoHub: https://github.com/MyoHub/myo_sim. Repository code licence: Apache-2.0. The project modifies morphology registration, helper inertias, coordinate reduction and contact mechanics; this is not an unmodified reference model.
- **MuJoCo** provides rigid-body kinematics and dynamics; the project also independently assembles Newton–Euler wrenches and uses a custom conditional muscle-force-sharing calculation. OpenSim is used for imported shoulder path geometry, not as the sole full-body dynamics solver.
- Shoulder source: **Seth et al. (2019)**, *Muscle Contributions to Upper-Extremity Movement and Work From a Musculoskeletal Model of the Human Shoulder*, https://doi.org/10.3389/fnbot.2019.00090. Model: https://simtk.org/projects/thoracoscapular.
- Imported RMR model data are adapted from the shoulder source. **Belli, Joshi, Prendergast, Beck, dell Santina, Peternel & Seth**, copyright 2023, CC BY 4.0; retained source notice: `models/rmr/LICENSE_data`. The project registers and mirrors shoulder geometry and implements conservative derivatives of its wrapping approximation.
- CMJ motion/data: **Calisti, Mohr & Federolf (2025)**, https://doi.org/10.1038/s41597-025-05934-5; dataset https://doi.org/10.6084/m9.figshare.28890545, CC BY 4.0. Retargeting, force analysis and rendering are project derivatives. The measured dataset does not provide our individual muscle forces.
- Squat muscle-force comparison: **Goodman, Helms & Graham (2023)**, https://doi.org/10.1519/JSC.0000000000004506, Table 1. These are literature estimates from a different protocol and participants.
- Loaded-squat reference: **Pürzel et al. (2025)**, https://doi.org/10.1111/sms.70058. Qualitative reference here; numerical full-text muscle-force curves were not extracted in this review.
- CMJ tendon-force comparison: **Wade, Lichtwark & Farris (2018)**, https://doi.org/10.1038/s41598-018-20387-0. Reference Achilles estimates are not identical to our scalar sum of calf-muscle tensions.

The final three-second cards cite the calculation model, case-relevant motion/force references and adapted anatomical assets. The complete per-component anatomical notices above remain applicable. No reference is presented as an independent validation of this project's personal muscle forces. Credit frames are excluded from all motion statistics.


## Source preservation

Full preserved model notices and experimental code: https://github.com/tmdals1213-cyber/exercise-mechanics-lab/tree/lab-source/docs/upstream_notices and https://github.com/tmdals1213-cyber/exercise-mechanics-lab/tree/lab-source. No new license has been selected for original project code. Preserve the component-specific anatomy notices and applicable licences when redistributing derived material.
