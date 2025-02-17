# WorldPose README (Ver 1.0)
## Note
As mentioned on the website, this dataset includes only the camera and pose data. The broadcast footage is owned by FIFA and requires a separate agreement. Please refer to the email regarding the application process for video data.

## Evaluation
We perform evaluations on the following clips:

```
ARG_FRA_182345
ARG_FRA_201902
BRA_KOR_231503
CRO_MOR_180400
FRA_MOR_231753
NET_ARG_231259
```

## Data Format
### Cameras
```python
# Assuming N players and T frames
{
  "K": K, # np.array of shape (T, 3, 3), intrinsic matrix
  "R": R, # np.array of shape (T, 3, 3), rotation matrix
  "t": t, # np.array of shape (T, 3), translation vector
  "k": k, # np.array of shape (T, 5), distortion coefficients (k1, k2, p1, p2, k3).
}
```

### Poses
```python
# Assuming N players and T frames
# If a player is not visible in a particular frame, the corresponding data will be set to NaN
{
  "betas": betas,                   # np.array of (N, 10)
  "global_orients": global_orients, # np.array of (N, T, 3)
  "body_poses": body_poses,         # np.array of (N, T, 69)
  "transl": transl,                 # np.array of (N, T, 3)
}
```
