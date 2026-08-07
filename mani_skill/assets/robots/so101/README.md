# SO101

This is the 6DOF SO101 arm, the successor to the SO100. The original assets are from [here](https://github.com/TheRobotStudio/SO-ARM100) and were provided under a [Apache 2.0 License](LICENSE)

Source files: `Simulation/SO101/so101_new_calib.urdf` and the 13 STL meshes
under `Simulation/SO101/assets/`.

We use the `new_calib` URDF rather than `old_calib` because its joint limits are
centred on zero (`shoulder_lift` `[-1.745, 1.745]`, `elbow_flex` `[-1.69, 1.69]`)
rather than offset (`[-3.316, 0.175]`, `[-0.175, 3.142]`). A zero-centred
calibration makes the all-zeros qpos a physically meaningful right-angle home
pose, which matches how the real arm is aligned during setup.

Changes made:
- renamed the robot from `so101_new_calib` to `so101`
- moved meshes into `meshes/` and rewrote the URDF's mesh paths to match
- decimated the visual meshes from 322,564 to 129,614 triangles. Each mesh uses
  the most aggressive quadric decimation that keeps its volume within 3% and its
  bounding box within 1mm of the original, so no part visibly grows or shrinks
- added the `.convex.stl` convex-hull caches SAPIEN would otherwise generate on
  first load
- manual decomposition of both gripper link collision meshes into simpler
  meshes: `moving_jaw_so101_v1` and `wrist_roll_follower_so101_v1` are each
  described by 6 convex `.ply` pieces. A single convex hull of either mesh would
  fill in the gripper's notch (2.33x and 2.58x the true volume respectively) and
  make grasping impossible; the decomposition brings that to 1.13x and 1.52x
- added `so101.srdf` disabling collisions between adjacent links and between
  pairs that cannot collide given the joint limits
- added a `moving_jaw_tip` frame link at the moving jaw's distal face. The
  source URDF already provides `gripper_frame_link` at the *fixed* jaw's tip,
  so the `SO101` agent averages the two to place the TCP at the grasp point --
  the same construction the so100 uses. This matters beyond convenience:
  `PickCubeEnv` measures `tcp_to_obj_dist` from `agent.tcp_pos`, so a TCP on
  one jaw rather than between them would miscalibrate the task reward
- dropped the `<inertial>` on the geometry-less frame links. The source gives
  them `mass=1e-9` with an all-zero inertia tensor, and PhysX crashes on that
  degenerate rigid body; the so100 URDF likewise carries no inertial on its
  tip links

Left untouched:
- link masses and inertia tensors, which are copied verbatim from the source
  URDF. They are already realistic (1e-5 to 1e-8 range, with off-diagonal
  terms), so no values were rounded or zeroed
- joint names, types, limits, origins and axes

Note for anyone writing controllers against this arm: `gripper_frame_link` is
rotated 180 degrees about y relative to its parent `gripper_link`, and the
gripper joint runs `[-0.1745, 1.7453]` from closed to open -- the opposite
sense to the so100, whose gripper closes toward negative values from an open 0.
