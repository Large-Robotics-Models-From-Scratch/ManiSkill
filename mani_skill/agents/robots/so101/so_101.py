import copy

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/so101/so101.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link={
            "gripper_link": dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            "moving_jaw_so101_v1_link": dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    # The gripper joint runs [-0.1745, 1.7453]: the lower limit is fully
    # closed and the upper limit fully open. Note this is the opposite sense
    # to the SO100, whose gripper closes toward negative from an open 0.
    GRIPPER_CLOSED = -0.1745
    GRIPPER_OPEN = 1.7453

    # Rotor inertia of the STS3215 servos, from TheRobotStudio's
    # Simulation/SO101/joints_properties.xml. ManiSkill leaves armature at 0,
    # which is fine for heavier arms, but the so101's links are light enough
    # that a stiff position controller becomes numerically unstable without
    # it: the arm chatters at ~1 rad/s, never satisfies is_robot_static, and
    # cannot hold a placement. Adding the real rotor inertia settles it to
    # ~0.11 rad/s.
    ARMATURE = 0.028

    # Every arm joint sits at a multiple of pi/2 so the pose can be
    # reproduced on the real arm by eye -- the new_calib URDF is centred on
    # zero, which makes right angles physically meaningful here.
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, 0, GRIPPER_CLOSED]),
            pose=sapien.Pose(),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        # max delta permitted of 0.05 since the robot is not as accurate as more expensive arms
        # and moving too fast can cause the robot to shake too much and damage the hardware
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        # gripper_frame_link ships with the source URDF and sits at the
        # fixed jaw's tip; moving_jaw_tip is its counterpart on the moving
        # jaw. The TCP is their midpoint, so it lands at the grasp point
        # rather than on either jaw.
        self.finger1_tip = self.robot.links_map["gripper_frame_link"]
        self.finger2_tip = self.robot.links_map["moving_jaw_tip"]
        # Arm joints only. The gripper needs to stay responsive: giving it
        # rotor inertia too makes it close so sluggishly that the jaws never
        # secure the cube before the lift begins.
        for articulation in self.robot._objs:
            for joint in articulation.get_active_joints():
                if joint.name.endswith("gripper"):
                    continue
                joint.armature = np.full(
                    np.shape(joint.armature), self.ARMATURE, dtype=np.float32
                )

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (gripper_frame_link)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
