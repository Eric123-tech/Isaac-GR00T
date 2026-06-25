# ManipTrans Assets for OakInk2 Retargeting

This directory stores local copies of the minimum ManipTrans assets needed for
OakInk2 to Artimano retargeting inside the GR00T repository.

Expected local layout:

```text
third_party/maniptrans_assets/
  smplx/
    SMPLX_NEUTRAL.npz
    version.txt
  smplx_extra/
    body_upper_idx.pt
  mano_urdf/
    rh_mano.urdf
    lh_mano.urdf
    rh_urdf_meshes/
    lh_urdf_meshes/
```

Source paths in the local ManipTrans checkout:

```text
ManipTrans/data/body_utils/body_models/smplx/SMPLX_NEUTRAL.npz
ManipTrans/data/body_utils/body_models/smplx/version.txt
ManipTrans/data/smplx_extra/body_upper_idx.pt
ManipTrans/maniptrans_envs/assets/mano_urdf/rh_mano.urdf
ManipTrans/maniptrans_envs/assets/mano_urdf/lh_mano.urdf
ManipTrans/maniptrans_envs/assets/mano_urdf/rh_urdf_meshes/
ManipTrans/maniptrans_envs/assets/mano_urdf/lh_urdf_meshes/
```

The asset files are intentionally ignored by Git in this directory. Do not push
SMPL-X model files or copied ManipTrans mesh assets unless licensing and Git LFS
handling are explicitly resolved.
