#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np

from aruco_opencv_msgs.msg import ArucoDetection
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# ============================
# Temporal filter parameters
# ============================

ALPHA_POS = 0.4    # 0.2–0.5 recomendado
ALPHA_YAW = 0.3

# ============================
# Fixed ArUco map (world_ned)
# ============================

CIRTESU_MESH_PATH = "package://girona500_description/meshes/cirtesu.dae"
SIGMA_DIST = 1.0 

CIRTESU_MESH_SCALE = [1.0, 1.0, 1.0]

# Pose relativa al frame cirtesu_base_link
CIRTESU_MESH_POS = [0.0, 0.0, 0.2]

# Rotación mesh (yaw normalmente necesario)
CIRTESU_MESH_YAW = 0.0

ARUCO_YAW_OFFSET = -np.pi/2 

ARUCO_MAP = {
    0: (-1.35, 3.0, 5.0),
    1: (-4.35, -3.0, 5.0),
    2: ( 1.65,  0.0, 5.0),
    3: ( 1.65,  3.0, 5.0),
    4: (-1.35,  0.0, 5.0),
    5: (-4.35,  3.0, 5.0),
    6: (-4.35,  0.0, 5.0),
    7: (-1.35, -3.0, 5.0),
    8: ( 1.65, -3.0, 5.0),
}


class ArucoMapLocalization:

    def __init__(self):

        # Temporal filter memory
        self.prev_pos = None
        self.prev_yaw = None

        rospy.init_node("aruco_map_localization")

        self.world_frame  = "world_ned"
        self.base_frame   = "girona500/base_link"
        self.camera_frame = "girona500/down_camera/camera"

        # Topics
        self.aruco_topic  = "/girona500/down_camera/aruco_detections"
        self.marker_topic = "/tandem_girona/aruco_map_markers"
        self.pose_topic   = "/girona500/navigator/aruco_pose"

        # TF
        self.tf_buffer   = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.marker_pub = rospy.Publisher(
            self.marker_topic,
            MarkerArray,
            queue_size=1
        )

        self.pose_pub = rospy.Publisher(
            self.pose_topic,
            PoseWithCovarianceStamped,
            queue_size=1
        )

        # Subscriber
        rospy.Subscriber(
            self.aruco_topic,
            ArucoDetection,
            self.aruco_callback,
            queue_size=1
        )

        rospy.loginfo("Aruco map localization node started")

    # =====================================================
    # MAIN CALLBACK
    # =====================================================

    def aruco_callback(self, msg):

        visible_ids = [m.marker_id for m in msg.markers]

        # -------------------------------------------------
        # 1) Publish fixed map MarkerArray
        # -------------------------------------------------

        marker_array = MarkerArray()

        mesh_marker = Marker()

        mesh_marker.header.frame_id = "cirtesu_base_link"
        mesh_marker.header.stamp = rospy.Time.now()

        mesh_marker.ns = "cirtesu_mesh"
        mesh_marker.id = 1000   # ID alto para no colisionar con ArUcos

        mesh_marker.type = Marker.MESH_RESOURCE
        mesh_marker.action = Marker.ADD

        mesh_marker.mesh_resource = CIRTESU_MESH_PATH
        mesh_marker.mesh_use_embedded_materials = True

        mesh_marker.scale.x = CIRTESU_MESH_SCALE[0]
        mesh_marker.scale.y = CIRTESU_MESH_SCALE[1]
        mesh_marker.scale.z = CIRTESU_MESH_SCALE[2]

        mesh_marker.pose.position.x = CIRTESU_MESH_POS[0]
        mesh_marker.pose.position.y = CIRTESU_MESH_POS[1]
        mesh_marker.pose.position.z = CIRTESU_MESH_POS[2]

        qx, qy, qz, qw = quaternion_from_euler(3.1416, 0.0, CIRTESU_MESH_YAW)

        mesh_marker.pose.orientation.x = qx
        mesh_marker.pose.orientation.y = qy
        mesh_marker.pose.orientation.z = qz
        mesh_marker.pose.orientation.w = qw

        mesh_marker.color.a = 1.0   # obligatorio aunque use materiales

        marker_array.markers.append(mesh_marker)

        for mid, (mx, my, mz) in ARUCO_MAP.items():

            m = Marker()
            m.header.frame_id = "cirtesu_base_link"
            m.header.stamp = rospy.Time.now()

            m.ns = "aruco_map"
            m.id = mid
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = mx
            m.pose.position.y = my
            m.pose.position.z = mz
            m.pose.orientation.w = 1.0

            m.scale.x = 0.30
            m.scale.y = 0.30
            m.scale.z = 0.08

            # Default GREEN
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8

            # If visible -> YELLOW
            if mid in visible_ids:
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 0.0

            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

        # -------------------------------------------------
        # 2) Robot localization
        # -------------------------------------------------

        if len(msg.markers) == 0:
            return

        try:
            tf_base_cam = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(0.2)
            )
        except:
            rospy.logwarn_throttle(1.0, "TF base->camera not available")
            return

        estimates = []
        weights = []

        yaw_estimates = []
        yaw_weights = []

        for marker in msg.markers:

            mid = marker.marker_id

            if mid not in ARUCO_MAP:
                continue

            # Marker pose in camera frame
            cam_marker = PoseStamped()
            cam_marker.header = msg.header
            cam_marker.pose = marker.pose

            try:
                # camera -> base
                base_marker = tf2_geometry_msgs.do_transform_pose(
                    cam_marker,
                    tf_base_cam
                )
            except:
                continue

            # Known world marker position
            wx, wy, wz = ARUCO_MAP[mid]

            # Compute base pose in world (position)
            rx = wx - base_marker.pose.position.x
            ry = wy - base_marker.pose.position.y
            rz = wz - base_marker.pose.position.z

            # --------------------------------
            # Distance weight (closer = better)
            # --------------------------------

            dx = base_marker.pose.position.x
            dy = base_marker.pose.position.y
            dz = base_marker.pose.position.z

            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            weight = np.exp(-(dist**2) / (2.0 * SIGMA_DIST**2))

            # Store weighted pose
            estimates.append([rx, ry, rz])
            weights.append(weight)

            # --------------------------------
            # Orientation (yaw from ArUco)
            # --------------------------------

            q = base_marker.pose.orientation

            (_, _, yaw_marker) = euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )

            # Convert marker yaw -> robot yaw
            yaw_robot = yaw_marker + ARUCO_YAW_OFFSET

            # Normalize angle
            yaw_robot = np.arctan2(np.sin(yaw_robot), np.cos(yaw_robot))

            yaw_estimates.append(yaw_robot)
            yaw_weights.append(weight)

        if len(estimates) == 0:
            return

        est_np = np.array(estimates)
        w_np   = np.array(weights)

        mean_pos = np.sum(est_np * w_np[:, None], axis=0) / np.sum(w_np)

        yaw_np = np.array(yaw_estimates)
        wy_np  = np.array(yaw_weights)

        sin_sum = np.sum(np.sin(yaw_np) * wy_np)
        cos_sum = np.sum(np.cos(yaw_np) * wy_np)


        mean_yaw = np.arctan2(sin_sum, cos_sum)

        # =====================================================
        # Temporal filtering (EMA)
        # =====================================================

        if self.prev_pos is None:

            filt_pos = mean_pos
            filt_yaw = mean_yaw

        else:

            # Position EMA
            filt_pos = (
                ALPHA_POS * mean_pos +
                (1.0 - ALPHA_POS) * self.prev_pos
            )

            # Yaw circular EMA
            dyaw = mean_yaw - self.prev_yaw
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))

            filt_yaw = self.prev_yaw + ALPHA_YAW * dyaw

        # Store state
        self.prev_pos = filt_pos
        self.prev_yaw = filt_yaw


        # =====================================================
        # Build local pose (base_link)
        # =====================================================

        out = PoseWithCovarianceStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "cirtesu_base_link"

        out.pose.pose.position.x = filt_pos[0]
        out.pose.pose.position.y = filt_pos[1]
        out.pose.pose.position.z = filt_pos[2]

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -filt_yaw)

        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw


        # =====================================================
        # Covariance
        # =====================================================

        total_weight = np.sum(weights)

        sigma_xy  = 0.13  / np.sqrt(total_weight)
        sigma_z   = 0.005 / np.sqrt(total_weight)
        sigma_yaw = 1.0   / np.sqrt(total_weight)

        cov = [0.0]*36

        cov[0]  = sigma_xy     # x
        cov[7]  = sigma_xy     # y
        cov[14] = sigma_z      # z

        cov[21] = 0.5         
        cov[28] = 0.5         
        cov[35] = sigma_yaw   # yaw

        out.pose.covariance = cov


        # =====================================================
        # Transform pose to world_ned
        # =====================================================

        # Create PoseStamped for TF
        pose_local = PoseStamped()
        pose_local.header.stamp = out.header.stamp
        pose_local.header.frame_id = "cirtesu_base_link"
        pose_local.pose = out.pose.pose


        # Lookup TF
        try:
            tf_world_base = self.tf_buffer.lookup_transform(
                self.world_frame,        # "world_ned"
                "cirtesu_base_link",
                rospy.Time(0),
                rospy.Duration(0.2)
            )
        except:
            rospy.logwarn_throttle(1.0, "TF world_ned -> base not available")
            return


        # Transform pose
        try:
            pose_world = tf2_geometry_msgs.do_transform_pose(
                pose_local,
                tf_world_base
            )
        except:
            rospy.logwarn("Pose transform failed")
            return


        # =====================================================
        # Publish final pose in NED
        # =====================================================

        out.header.frame_id = "world_ned"
        out.pose.pose = pose_world.pose

        self.pose_pub.publish(out)

if __name__ == "__main__":
    try:
        ArucoMapLocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
