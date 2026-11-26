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
