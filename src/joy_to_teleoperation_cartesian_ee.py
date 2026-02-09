#!/usr/bin/env python3
# Copyright (c) 2017 Iqua Robotics SL - All Rights Reserved
#
# This file is subject to the terms and conditions defined in file
# 'LICENSE.txt', which is part of this source code package.


"""
@@>LogitechFX10Atlantis controler node.<@@
"""

import rospy
from cola2_control.joystickbase import JoystickBase
from cola2_ros import param_loader
from std_srvs.srv import Empty, Trigger, TriggerRequest, TriggerResponse
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64
from controller_manager_msgs.srv import SwitchController
from cola2_msgs.srv import DigitalOutput, String
from std_srvs.srv import SetBool
from geometry_msgs.msg import TwistStamped, PoseStamped
import tf2_ros
from iauv_kinematic_control.srv import SwitchTasks
class LogitechFX10Atlantis(JoystickBase):
    """LogitechFX10Atlantis controler node."""

    """
        This class inherent from JoystickBase. It has to overload the
        method update_joy(self, joy) that receives a sensor_msgs/Joy
        message and fill the var self.joy_msg as described in the class
        JoystickBase.
        From this class it is also possible to call services or anything
        else reading the buttons in the update_joy method.
    """

    # JOYSTICK  DEFINITION:
    LEFT_JOY_HORIZONTAL = 0  # LEFT+, RIGHT-
    LEFT_JOY_VERTICAL = 1  # UP+, DOWN-
    LEFT_TRIGGER = 2  # NOT PRESS 1, PRESS -1
    RIGHT_JOY_HORIZONTAL = 3  # LEFT+, RIGHT-
    RIGHT_JOY_VERTICAL = 4  # UP+, DOWN-
    RIGHT_TRIGGER = 5  # NOT PRESS 1, PRESS -1
    CROSS_HORIZONTAL = 6  # LEFT+, RIGHT-
    CROSS_VERTICAL = 7  # UP+, DOWN-
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LEFT = 4
    BUTTON_RIGHT = 5
    BUTTON_BACK = 6
    BUTTON_START = 7
    BUTTON_LOGITECH = 8
    BUTTON_LEFT_JOY = 9
    BUTTON_RIGHT_JOY = 10
    MOVE_UP = 1
    MOVE_DOWN = -1
    MOVE_LEFT = 1
    MOVE_RIGHT = -1

    def __init__(self, name):
        """Class constructor."""
        JoystickBase.__init__(self, name)
        rospy.loginfo("%s: LogitechFX10 constructor", name)
        self.req_enable_joint_pos = False
        self.req_disable_joint_pos = False

        # To select mode of operation
        self.mode = 0

        # To transform button into axis
        self.up_down = 0.0
        self.left_right = 0.0
        self.start_service = ""
        self.stop_service = ""
        self.joint_pos_task_active = False
        self.bravo_ee_admittance_task = "bravo_ee_admittance"
        self.bravo_ee_ff_task = "bravo_ee_configuration_feedforward"

        ik_cfg = rospy.get_param("~ik_mode")

        self.ik_world_frame = ik_cfg["frames"]["world"]
        self.cartesian_velocity_frame = ik_cfg["frames"]["cartesian_velocity_frame"]
        self.ik_ee_frame = ik_cfg["frames"]["ee"]

        self.scale_lin = ik_cfg["scaling"]["linear"]
        self.scale_ang = ik_cfg["scaling"]["angular"]
        self.deadband = ik_cfg["deadband"]["velocity_norm"]

        jp_cfg = rospy.get_param("~ik_mode/tasks/joint_positions")
        adm_cfg = rospy.get_param("~ik_mode/tasks/bravo_ee_admittance")
        ff_cfg  = rospy.get_param("~ik_mode/tasks/bravo_ee_configuration_feedforward")

        self.bravo_ee_admittance_task = adm_cfg["name"]
        self.bravo_ee_ff_task = ff_cfg["name"]

        self.joint_pos_task_name = jp_cfg["name"]

        joint_names = jp_cfg["joints"]["names"]
        joint_target = jp_cfg["joints"]["target"]

        # Seguridad básica
        if len(joint_names) != len(joint_target):
            rospy.logfatal(
                "joint_positions: names and target size mismatch "
                f"({len(joint_names)} vs {len(joint_target)})"
            )
            rospy.signal_shutdown("Invalid joint_positions config")

        namespace = rospy.get_namespace()
        self.get_config()

        # Create client to services:
        # ... start button service
        if self.start_service != "":
            done = False
            while not done and not rospy.is_shutdown():
                try:
                    rospy.wait_for_service(self.start_service, 10)
                    self.enable_keep_pose = rospy.ServiceProxy(self.start_service, Trigger)
                    done = True
                except rospy.ROSException as e:
                    rospy.logwarn("%s: Service call failed: %s", self.name, e)
                    rospy.sleep(1.0)

        # ... stop button service
        if self.stop_service != '':
            rospy.wait_for_service(self.stop_service, 10)
            try:
                self.disable_keep_pose = rospy.ServiceProxy(self.stop_service, Trigger)
            except rospy.ServiceException as e:
                rospy.logwarn("%s: Service call failed: %s", self.name, e)

        # ... enable thrusters service
        rospy.wait_for_service(
            namespace + 'teleoperation/enable_thrusters', 10)
        try:
            self.enable_thrusters = rospy.ServiceProxy(
                namespace + 'teleoperation/enable_thrusters', Trigger)
        except rospy.ServiceException as e:
            rospy.logwarn("%s: Service call failed: %s", self.name, e)

        # # ... disable thrusters service
        rospy.wait_for_service(
            namespace + 'teleoperation/disable_thrusters', 10)
        try:
            self.disable_thrusters = rospy.ServiceProxy(
                namespace + 'teleoperation/disable_thrusters', Trigger)
        except rospy.ServiceException as e:
            rospy.logwarn("%s: Service call failed: %s", self.name, e)

        namespace = rospy.get_namespace()
        self.get_config()

        # Create publisher to manipulator 
        bravo_ns = rospy.get_param("~bravo_ns")
        self.gripper_mode = rospy.get_param("~gripper_mode", "simulation")

        self.pub_bravo_ee_ff = rospy.Publisher(
            '/tp_controller/tasks/bravo_ee_configuration_feedforward/feedforward',
            TwistStamped,
            queue_size=1
        )
        self.pub_bravo_ee_target = rospy.Publisher(
            '/tp_controller/tasks/bravo_ee_configuration_feedforward/target',
            PoseStamped,
            queue_size=1
        )
        # TF listener for EE pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.ee_ff_cmd = TwistStamped()

        self.pub_bravoarm_desired_joint_velocity = rospy.Publisher(
            namespace + bravo_ns + '/joint_teleop_velocity_controller/command',
            Float64MultiArray,
            queue_size = 1)

        if self.gripper_mode == "simulation":
            self.pub_gripper_large = rospy.Publisher(
                namespace + bravo_ns + '/gripper_velocity_controller_finger_large/command',
                Float64,
                queue_size=1
            )
            self.pub_gripper_small = rospy.Publisher(
                namespace + bravo_ns + '/gripper_velocity_controller_finger_small/command',
                Float64,
                queue_size=1
            )
            self.gripper_scale = 0.2

            # ---- controllers names (simulation) ----
            self.gripper_vel_ctrls = [
                "gripper_velocity_controller_finger_large",
                "gripper_velocity_controller_finger_small"
            ]
            self.gripper_pos_ctrls = [
                "gripper_position_controller_finger_large",
                "gripper_position_controller_finger_small"
            ]
        else:  # REAL ROBOT
            self.pub_gripper_real = rospy.Publisher(
                namespace + bravo_ns + '/gripper_velocity_controller/command',
                Float64,
                queue_size=1
            )
            self.gripper_scale = 0.004

             # ---- controllers names (real robot) ----
            self.gripper_vel_ctrls = ["gripper_velocity_controller"]
            self.gripper_pos_ctrls = ["gripper_position_controller"]

        self.switch_tasks_srv = rospy.ServiceProxy(
            '/tp_controller/switch_tasks',
            SwitchTasks
        )
        self.pub_joint_pos_target = rospy.Publisher(
            '/tp_controller/tasks/joint_positions/target',
            Float64MultiArray,
            queue_size=1
        ) 
        #target de nominal position task
        self.joint_pos_target = Float64MultiArray()
        self.joint_pos_target.layout.dim = [MultiArrayDimension()]
        self.joint_pos_target.layout.dim[0].label = 'joints'
        self.joint_pos_target.layout.dim[0].size = len(joint_target)
        self.joint_pos_target.layout.dim[0].stride = 1
        self.joint_pos_target.layout.data_offset = 0
        self.joint_pos_target.data = joint_target

        self.bravoarm_joint_cmd = Float64MultiArray()
        self.bravoarm_joint_cmd.layout.dim = [MultiArrayDimension()]
        self.bravoarm_joint_cmd.layout.dim[0].label = "velocity"
        self.bravoarm_joint_cmd.layout.dim[0].size = 6
        self.bravoarm_joint_cmd.layout.dim[0].stride = 0
        self.bravoarm_joint_cmd.layout.data_offset = 0
        self.bravoarm_joint_cmd.data = [0.0]*6

        self.bravoarm_gripper_cmd = 0.0
        rospy.loginfo(
            f"[IK mode] joint_positions target for joints: {joint_names}"
        )   

    def switch_bravo_controllers(self, start_ctrls, stop_ctrls):
        srv = "/girona500/bravo/controller_manager/switch_controller"
        try:
            rospy.wait_for_service(srv, timeout=1.0)
            rospy.ServiceProxy(srv, SwitchController)(
                start_ctrls,
                stop_ctrls,
                1,
                True,
                0.0
            )
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn("SwitchController failed (%s): %s", srv, str(e))


    def update_joy(self, joy):
        """Receive joystic raw data."""
        """Transform FX10 joy data into 12 axis data (pose + twist)
        and sets the buttons that especify if position or velocity
        commands are used in the teleoperation."""
        self.mutual_exclusion.acquire()

        self.joy_msg.header = joy.header

        
        # Enable/disable keep position
        if joy.buttons[self.BUTTON_START] == 1.0:
            rospy.loginfo("%s: Start button service called", self.name)
            res = self.enable_keep_pose(TriggerRequest())
            if not res.success:
                rospy.logwarn("%s: Impossible to enable keep position, captain response: %s", self.name, res.message)
        if joy.buttons[self.BUTTON_BACK] == 1.0:
            rospy.loginfo("%s: Stop button service called", self.name)
            res = self.disable_keep_pose(TriggerRequest())
            if not res.success:
                rospy.logwarn("%s: Impossible to disable keep position, captain response: %s", self.name, res.message)

        # Enable/disable thrusters
        if joy.axes[self.LEFT_TRIGGER] < -0.9 and joy.axes[self.RIGHT_TRIGGER] < -0.9:
            rospy.loginfo("%s: DISABLE THRUSTERS!", self.name)
            self.disable_thrusters()
            rospy.sleep(1.0)

        if joy.buttons[self.BUTTON_LEFT] == 1.0 and joy.buttons[self.BUTTON_RIGHT] == 1.0:
            rospy.loginfo("%s: ENABLE THRUSTERS!", self.name)
            self.enable_thrusters()
        
        # Set zero desired joint velocities
        self.bravoarm_joint_cmd.data = [0.0]*6
        self.bravoarm_gripper_cmd = 0.0

        old_mode = self.mode
        if joy.buttons[self.BUTTON_LOGITECH] == 1:
            rospy.loginfo("Digital output of bravo turned off")
            rospy.ServiceProxy('/girona500/main_control_board/digital_output', DigitalOutput)(digital_output=1, value=0)

        if joy.buttons[self.BUTTON_RIGHT] == 1:
            if joy.buttons[self.BUTTON_X] == 1.0: # Left arm (rightarm)
                rospy.loginfo("Switching to Predefined positions")
                self.switch_bravo_controllers(
                    start_ctrls=["joint_trajectory_controller"],
                    stop_ctrls=["joint_teleop_velocity_controller", "joint_velocity_controller"] + self.gripper_vel_ctrls + self.gripper_pos_ctrls
                )

                try:
                    rospy.wait_for_service('/girona500/tp_controller/active', timeout=1.0)
                    rospy.ServiceProxy('/girona500/tp_controller/active', SetBool)(False)
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn("No se pudo llamar al servicio /girona500/tp_controller/active: %s", str(e))

                self.mode = 2 #predefined positions mode
            elif joy.buttons[self.BUTTON_B] == 1.0: # Right arm (leftarm)
                rospy.loginfo("Switching to bravo teleop controller")
                
                self.switch_bravo_controllers(
                    start_ctrls=["joint_teleop_velocity_controller"] + self.gripper_vel_ctrls,
                    stop_ctrls=["joint_velocity_controller", "joint_trajectory_controller"] + self.gripper_pos_ctrls
                )
                try:
                    rospy.wait_for_service('/girona500/tp_controller/active', timeout=1.0)
                    rospy.ServiceProxy('/girona500/tp_controller/active', SetBool)(False)
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn("No se pudo llamar al servicio /girona500/tp_controller/active: %s", str(e))

                self.mode = 1 #bravoarm
            elif joy.buttons[self.BUTTON_A] == 1.0: # AUV
                rospy.loginfo("Switching to AUV control")
                self.mode = 0
                try:
                    rospy.wait_for_service('/girona500/tp_controller/active', timeout=1.0)
                    rospy.ServiceProxy('/girona500/tp_controller/active', SetBool)(False)
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn("No se pudo llamar al servicio /girona500/tp_controller/active: %s", str(e))
            elif (joy.axes[self.CROSS_HORIZONTAL] == self.MOVE_RIGHT):
                rospy.loginfo("Switching to bravo ik controller cartesian velocity ff teleop")
                self.mode = 3 #bravo ik mode (cartesian velocity ff teleop)

                self.switch_bravo_controllers(
                    start_ctrls=["joint_velocity_controller"] + self.gripper_vel_ctrls,
                    stop_ctrls=["joint_teleop_velocity_controller", "joint_trajectory_controller"] + self.gripper_pos_ctrls
                )
                try:
                    rospy.wait_for_service('/girona500/tp_controller/active', timeout=1.0)
                    rospy.ServiceProxy('/girona500/tp_controller/active', SetBool)(True)
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn("No se pudo llamar al servicio /girona500/tp_controller/active: %s", str(e))
                try:
                    rospy.wait_for_service('/tp_controller/switch_tasks', timeout=0.2)

                    self.switch_tasks_srv(
                        enable_tasks=[self.bravo_ee_ff_task],
                        disable_tasks=[self.bravo_ee_admittance_task]
                    )

                    rospy.loginfo("[IAUV AUTO] Feedforward enabled, Admittance disabled")

                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn_throttle(1.0, f"[IAUV AUTO] Switch failed: {e}")

            elif (joy.axes[self.CROSS_VERTICAL] == self.MOVE_UP):
                rospy.loginfo("Switching to bravo ik controller admitance teleop")
                self.mode = 3 #bravo ik mode (cartesian velocity admitance)

                self.switch_bravo_controllers(
                    start_ctrls=["joint_velocity_controller"] + self.gripper_vel_ctrls,
                    stop_ctrls=["joint_teleop_velocity_controller", "joint_trajectory_controller"] + self.gripper_pos_ctrls
                )
                try:
                    rospy.wait_for_service('/girona500/tp_controller/active', timeout=1.0)
                    rospy.ServiceProxy('/girona500/tp_controller/active', SetBool)(True)
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn("No se pudo llamar al servicio /girona500/tp_controller/active: %s", str(e))

                try:
                    rospy.wait_for_service('/tp_controller/switch_tasks', timeout=0.2)

                    self.switch_tasks_srv(
                        enable_tasks=[self.bravo_ee_admittance_task],
                        disable_tasks=[self.bravo_ee_ff_task]
                    )

                    rospy.loginfo("[IAUV AUTO] Admittance enabled, Feedforward disabled")

                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn_throttle(1.0, f"[IAUV AUTO] Switch failed: {e}")

        if old_mode != self.mode: #Mode changed: send zeros once
            if old_mode == 1: #leftarm
                self.pub_bravoarm_desired_joint_velocity.publish(self.bravoarm_joint_cmd)
                if self.gripper_mode == "simulation":
                    self.pub_gripper_large.publish(0.0)
                    self.pub_gripper_small.publish(0.0)
                else:
                    self.pub_gripper_real.publish(0.0)

        if self.mode == 0: #AUV mode

            # enable/disable z control position
            if joy.buttons[self.BUTTON_RIGHT] == 0: # not pushed
                self.joy_msg.buttons[JoystickBase.BUTTON_POSE_Z] = joy.buttons[self.BUTTON_A]
                self.joy_msg.buttons[JoystickBase.BUTTON_TWIST_W] = joy.buttons[self.BUTTON_Y]
                if joy.buttons[self.BUTTON_A] == 1.0:
                    self.up_down = 0.0
                    rospy.loginfo("%s: Reset up_down counter", self.name)
                # enable/disable yaw control position
                self.joy_msg.buttons[JoystickBase.BUTTON_POSE_YAW] = joy.buttons[self.BUTTON_B]
                self.joy_msg.buttons[JoystickBase.BUTTON_TWIST_R] = joy.buttons[self.BUTTON_X]
                if joy.buttons[self.BUTTON_B] == 1.0:
                    self.left_right = 0.0
                    rospy.loginfo("%s: Reset left_right counter", self.name)

            # Transform discrete axis (cross) to two 'analog' axis to control
            # depth and yaw in position.

            # up-down (depth control pose)
            if (joy.axes[self.CROSS_VERTICAL] == self.MOVE_DOWN):
                self.up_down = self.up_down + 0.05
                if self.up_down > 1.0:
                    self.up_down = 1.0
            elif (joy.axes[self.CROSS_VERTICAL] == self.MOVE_UP):
                self.up_down = self.up_down - 0.05
                if self.up_down < -1.0:
                    self.up_down = -1.0

            # left-right (yaw control pose)
            if (joy.axes[self.CROSS_HORIZONTAL] == self.MOVE_RIGHT):
                self.left_right = self.left_right + 0.05
                if self.left_right > 1.0:
                    self.left_right = -1.0
            elif (joy.axes[self.CROSS_HORIZONTAL] == self.MOVE_LEFT):
                self.left_right = self.left_right - 0.05
                if self.left_right < -1.0:
                    self.left_right = 1.0

            self.joy_msg.axes[JoystickBase.AXIS_POSE_Z] = self.up_down
            self.joy_msg.axes[JoystickBase.AXIS_POSE_YAW] = self.left_right
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_U] = joy.axes[self.RIGHT_JOY_VERTICAL]
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_V] = -joy.axes[self.RIGHT_JOY_HORIZONTAL]
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_W] = -joy.axes[self.LEFT_JOY_VERTICAL]
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_R] = -joy.axes[self.LEFT_JOY_HORIZONTAL]

        elif self.mode == 1: #bravoarm mode
            # Set zero desired AUV velocity and last pose
            self.joy_msg.axes[JoystickBase.AXIS_POSE_Z] = self.up_down
            self.joy_msg.axes[JoystickBase.AXIS_POSE_YAW] = self.left_right
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_U] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_V] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_W] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_R] = 0.0
            
            if joy.buttons[self.BUTTON_LEFT] == 0:
                self.bravoarm_joint_cmd.data[0] = joy.axes[self.LEFT_JOY_HORIZONTAL]*0.2
                self.bravoarm_joint_cmd.data[1] = -joy.axes[self.LEFT_JOY_VERTICAL]*0.2
            else:
                self.bravoarm_joint_cmd.data[5] = joy.axes[self.LEFT_JOY_HORIZONTAL]*0.5
                self.bravoarm_joint_cmd.data[4] = joy.axes[self.LEFT_JOY_VERTICAL]*0.2
            
            self.bravoarm_joint_cmd.data[2] = -joy.axes[self.RIGHT_JOY_VERTICAL]*0.2
            self.bravoarm_joint_cmd.data[3] = -joy.axes[self.RIGHT_JOY_HORIZONTAL]*0.2

            # Open/Close gripper
            gripper_velocity = self.gripper_scale
            if joy.axes[self.RIGHT_TRIGGER] < 0.95:
                self.bravoarm_gripper_cmd = (2.0 -(joy.axes[self.RIGHT_TRIGGER] + 1.0))*gripper_velocity
            elif joy.axes[self.LEFT_TRIGGER] < 0.95:
                self.bravoarm_gripper_cmd = -(2.0 -(joy.axes[self.LEFT_TRIGGER] + 1.0))*gripper_velocity
            else:
                self.bravoarm_gripper_cmd = 0.0
        elif self.mode == 2:
            if joy.buttons[self.BUTTON_LEFT_JOY] == 1.0:
                rospy.loginfo("Calling /girona500/bravo/predefined_positions/do data= 'home'")
                rospy.ServiceProxy('/girona500/bravo/predefined_positions/do', String)(data='home')
            if joy.buttons[self.BUTTON_RIGHT_JOY] == 1.0:
                rospy.loginfo("Calling /girona500/bravo/predefined_positions/do data= 'look_down'")
                rospy.ServiceProxy('/girona500/bravo/predefined_positions/do', String)(data='look_down')

        elif self.mode == 3: #bravo ik mode (cartesian velocity)
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_U] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_V] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_W] = 0.0
            self.joy_msg.axes[JoystickBase.AXIS_TWIST_R] = 0.0

            self.ee_ff_cmd.header.stamp = rospy.Time.now()
            self.ee_ff_cmd.header.frame_id = self.cartesian_velocity_frame
            
             # ---------- LEFT STICK ----------
            if joy.buttons[self.BUTTON_LEFT] == 0:
                # Traslación EE (X, Y)
                self.ee_ff_cmd.twist.linear.z =  self.scale_lin * -joy.axes[self.LEFT_JOY_VERTICAL]
                self.ee_ff_cmd.twist.angular.z = self.scale_lin * -joy.axes[self.LEFT_JOY_HORIZONTAL]
            else:
                # Rotación EE (X, Y)
                self.ee_ff_cmd.twist.angular.y =  self.scale_ang * joy.axes[self.LEFT_JOY_VERTICAL]
                self.ee_ff_cmd.twist.angular.x =  -self.scale_ang * joy.axes[self.LEFT_JOY_HORIZONTAL]

            # ---------- RIGHT STICK ----------
            # Z lineal
            self.ee_ff_cmd.twist.linear.x = self.scale_lin * joy.axes[self.RIGHT_JOY_VERTICAL]

            # Yaw EE
            self.ee_ff_cmd.twist.linear.y = -self.scale_ang * joy.axes[self.RIGHT_JOY_HORIZONTAL]

            # Open/Close gripper
            gripper_velocity = self.gripper_scale
            if joy.axes[self.RIGHT_TRIGGER] < 0.95:
                self.bravoarm_gripper_cmd = (2.0 -(joy.axes[self.RIGHT_TRIGGER] + 1.0))*gripper_velocity
            elif joy.axes[self.LEFT_TRIGGER] < 0.95:
                self.bravoarm_gripper_cmd = -(2.0 -(joy.axes[self.LEFT_TRIGGER] + 1.0))*gripper_velocity
            else:
                self.bravoarm_gripper_cmd = 0.0
            
            self.publish_current_ee_pose_as_target()

            # ---------- Activate an deactivate the nominal task ----------
            deadband = 0.05

            v_norm = abs(self.ee_ff_cmd.twist.linear.x) + \
                    abs(self.ee_ff_cmd.twist.linear.y) + \
                    abs(self.ee_ff_cmd.twist.linear.z) + \
                    abs(self.ee_ff_cmd.twist.angular.x) + \
                    abs(self.ee_ff_cmd.twist.angular.y) + \
                    abs(self.ee_ff_cmd.twist.angular.z)
            if v_norm < deadband:
                if not self.joint_pos_task_active:
                    rospy.loginfo("[IK mode] Request enable joint_positions")

                    self.req_enable_joint_pos = True

                    # Publicar target UNA VEZ
                    self.pub_joint_pos_target.publish(self.joint_pos_target)

                    self.joint_pos_task_active = True
            else:
                # MOVIENDO → desactivar nominal
                if self.joint_pos_task_active:
                    rospy.loginfo("[IK mode] Request disable joint_positions")

                    self.req_disable_joint_pos = True

                    self.joint_pos_task_active = False

        self.mutual_exclusion.release()

    def publish_current_ee_pose_as_target(self):
        try:
            T = self.tf_buffer.lookup_transform(
                self.ik_world_frame,
                self.ik_ee_frame,
                rospy.Time(0),
                rospy.Duration(0.05)
            )

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.ik_world_frame
            pose.pose.position.x = T.transform.translation.x
            pose.pose.position.y = T.transform.translation.y
            pose.pose.position.z = T.transform.translation.z
            pose.pose.orientation = T.transform.rotation

            self.pub_bravo_ee_target.publish(pose)

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[IK mode] Cannot publish EE target: {e}")
            
    def iterate(self, event):
        """ This method is a callback of a timer. This is used to publish the
            output joy message """
        
        self.mutual_exclusion.acquire()
        # Publish message
        self.pub_map_ack_data.publish(self.joy_msg)
        if self.mode == 1: #leftarm
            self.pub_bravoarm_desired_joint_velocity.publish(self.bravoarm_joint_cmd)
            if self.gripper_mode == "simulation":
                self.pub_gripper_large.publish(-self.bravoarm_gripper_cmd)
                self.pub_gripper_small.publish(self.bravoarm_gripper_cmd)
            else:
                self.pub_gripper_real.publish(self.bravoarm_gripper_cmd)
        if self.mode == 3:
            self.pub_bravo_ee_ff.publish(self.ee_ff_cmd)
            if self.gripper_mode == "simulation":
                self.pub_gripper_large.publish(-self.bravoarm_gripper_cmd)
                self.pub_gripper_small.publish(self.bravoarm_gripper_cmd)
            else:
                self.pub_gripper_real.publish(self.bravoarm_gripper_cmd)

        self.mutual_exclusion.release()
        # Reset buttons
        # ---------- Handle TP switch outside joystick callback ----------
        if self.req_enable_joint_pos:

            try:
                rospy.wait_for_service('/tp_controller/switch_tasks', timeout=0.2)

                self.switch_tasks_srv(
                    enable_tasks=[self.joint_pos_task_name],
                    disable_tasks=[]
                )

                rospy.loginfo("[IK mode] joint_positions enabled")

            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn_throttle(1.0, f"[IK mode] Enable failed: {e}")

            self.req_enable_joint_pos = False


        if self.req_disable_joint_pos:

            try:
                rospy.wait_for_service('/tp_controller/switch_tasks', timeout=0.2)

                self.switch_tasks_srv(
                    enable_tasks=[],
                    disable_tasks=[self.joint_pos_task_name]
                )

                rospy.loginfo("[IK mode] joint_positions disabled")

            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn_throttle(1.0, f"[IK mode] Disable failed: {e}")

            self.req_disable_joint_pos = False


    def get_config(self):
        """ Read parameters from ROS Param Server """

        ns = rospy.get_namespace()

        param_dict = {'start_service': ('start_service', ''),
                      'stop_service': ('stop_service', '')
                     }

        param_loader.get_ros_params(self, param_dict)

if __name__ == '__main__':
    """ Initialize the logitech_fx10 node. """
    try:
        rospy.init_node('logitech_fx10_to_teleoperation')
        map_ack = LogitechFX10Atlantis(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
