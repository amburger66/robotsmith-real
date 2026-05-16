import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0

config_path = "configs"

robot_interface = FrankaInterface(
    f"{config_path}/charmander.yml",
    use_visualizer=False
)

controller_type = "JOINT_POSITION"
controller_cfg = YamlConfig(
    f"{config_path}/joint-position-controller.yml"
).as_easydict()

# These are home joints:
target_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

while True:

    if len(robot_interface._state_buffer) > 0:

        if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_joint_positions))) < 2e-3:
            break

    robot_interface.control(
        controller_type=controller_type,
        action=target_joint_positions + [OPEN_GRIPPER],
        controller_cfg=controller_cfg,
    )

robot_interface.close()
