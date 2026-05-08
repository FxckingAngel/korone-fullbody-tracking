import time
from abc import ABC, abstractmethod
from helpers import  sendToSteamVR, safe_normalize
from scipy.spatial.transform import Rotation as R
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
from pythonosc import udp_client

from helpers import shutdown
import numpy as np

class Backend(ABC):

    @abstractmethod
    def onparamchanged(self, params):
        ...

    @abstractmethod
    def connect(self, params):
        ...

    @abstractmethod
    def updatepose(self, params, pose3d, rots, hand_rots, visibility=None):
        ...

    @abstractmethod
    def disconnect(self):
        ...

class DummyBackend(Backend):

    def __init__(self, **kwargs):
        pass

    def onparamchanged(self, params):
        pass

    def connect(self, params):
        pass

    def updatepose(self, params, pose3d, rots, hand_rots, visibility=None):
        pass

    def disconnect(self):
        pass

class SteamVRBackend(Backend):

    def __init__(self, **kwargs):
        pass

    def onparamchanged(self, params):
        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)
            
    def connect(self, params):
        print("Connecting to SteamVR")

        #ask the driver, how many devices are connected to ensure we dont add additional trackers
        #in case we restart the program
        numtrackers = sendToSteamVR("numtrackers")
        if numtrackers is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

        numtrackers = int(numtrackers[2])

        # Default tracking now includes waist, feet, chest, and knees.
        totaltrackers = 23 if params.preview_skeleton else 6

        roles = [
            "TrackerRole_Waist",
            "TrackerRole_RightFoot",
            "TrackerRole_LeftFoot",
            "TrackerRole_Chest",
            "TrackerRole_RightKnee",
            "TrackerRole_LeftKnee",
        ]

        if params.ignore_hip and not params.preview_skeleton:
            del roles[0]
            totaltrackers -= 1

        if params.use_hands:
            totaltrackers += 2
            roles.append("TrackerRole_Handed")
            roles.append("TrackerRole_Handed")

        for i in range(len(roles),totaltrackers):
            roles.append("None")

        for i in range(numtrackers,totaltrackers):
            #sending addtracker to our driver will... add a tracker. to our driver.
            resp = sendToSteamVR(f"addtracker MPTracker{i} {roles[i]}")
            if resp is None:
                print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
                shutdown(params)

        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

    def updatepose(self, params, pose3d, rots, hand_rots, visibility=None):
        array = sendToSteamVR("getdevicepose 0")        #get hmd data to allign our skeleton to

        if array is None or len(array) < 10:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

        headsetpos = [float(array[3]),float(array[4]),float(array[5])]
        headsetrot = R.from_quat([float(array[7]),float(array[8]),float(array[9]),float(array[6])])

        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.

        if params.recalibrate:
            print("INFO: frame to recalibrate")

        else:
            pose3d = pose3d * params.posescale     #rescale skeleton to calibrated height
            #print(pose3d)
            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            if not params.preview_skeleton:
                tracker_payloads = []
                if not params.ignore_hip:
                    tracker_payloads.append((6, rots[0]))

                tracker_payloads.extend([
                    (0, rots[1]),
                    (5, rots[2]),
                    (7, build_chest_rotation(pose3d)),
                ])

                hip_span = pose3d[3] - pose3d[2]
                tracker_payloads.extend([
                    (4, build_knee_rotation(pose3d[3], pose3d[4], pose3d[5], hip_span)),
                    (1, build_knee_rotation(pose3d[2], pose3d[1], pose3d[0], -hip_span)),
                ])

                for tracker_idx, (joint_idx, rotation) in enumerate(tracker_payloads):
                    joint = pose3d[joint_idx] - offset
                    sendToSteamVR(
                        f"updatepose {tracker_idx} {joint[0]} {joint[1]} {joint[2]} "
                        f"{rotation[3]} {rotation[0]} {rotation[1]} {rotation[2]} {params.camera_latency} 0.8"
                    )

                numadded = len(tracker_payloads)
                if params.use_hands:
                    for i in [(10,0),(15,1)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]+numadded} {joint[0]} {joint[1]} {joint[2]} {hand_rots[i[1]][3]} {hand_rots[i[1]][0]} {hand_rots[i[1]][1]} {hand_rots[i[1]][2]} {params.camera_latency} 0.8")
            else:
                for i in range(23):
                    joint = pose3d[i] - offset      #if previewing skeleton, send the position of each keypoint to steamvr without rotation
                    sendToSteamVR(f"updatepose {i} {joint[0]} {joint[1]} {joint[2] - 2} 1 0 0 0 {params.camera_latency} 0.8")
        return True

    def disconnect(self):
        pass

def osc_build_msg(name, position_or_rotation, args):
    builder = osc_message_builder.OscMessageBuilder(address=f"/tracking/trackers/{name}/{position_or_rotation}")
    builder.add_arg(float(args[0]))
    builder.add_arg(float(args[1]))
    builder.add_arg(float(args[2]))
    return builder.build()


def visibility_average(visibility, indices):
    if visibility is None:
        return 1.0
    values = [float(visibility[idx]) for idx in indices]
    return float(np.mean(values)) if values else 1.0


def pose_to_vrchat_position(position):
    converted = np.array(position, dtype=float).copy()
    converted[2] = -converted[2]
    return converted


def quat_to_vrchat_euler(rotation):
    euler = R.from_quat(rotation).as_euler("zxy", degrees=True)
    return np.array([-euler[1], -euler[2], euler[0]], dtype=float)


def build_direction_rotation(forward, up_hint, right_hint):
    y_axis = safe_normalize(forward, fallback=np.array([0.0, 1.0, 0.0]))
    z_axis = np.cross(right_hint, y_axis)
    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.cross(up_hint, y_axis)
    z_axis = safe_normalize(z_axis, fallback=np.array([0.0, 0.0, 1.0]))
    x_axis = safe_normalize(np.cross(y_axis, z_axis), fallback=np.array([1.0, 0.0, 0.0]))
    z_axis = safe_normalize(np.cross(x_axis, y_axis), fallback=np.array([0.0, 0.0, 1.0]))
    return R.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T).as_quat()


def build_chest_rotation(pose3d):
    shoulder_axis = pose3d[13] - pose3d[12]
    spine_axis = pose3d[7] - pose3d[6]
    forward_hint = np.cross(shoulder_axis, spine_axis)
    if np.linalg.norm(forward_hint) < 1e-6:
        forward_hint = np.array([0.0, 0.0, 1.0])
    chest_basis = np.vstack((
        safe_normalize(shoulder_axis, fallback=np.array([1.0, 0.0, 0.0])),
        safe_normalize(spine_axis, fallback=np.array([0.0, 1.0, 0.0])),
        safe_normalize(forward_hint, fallback=np.array([0.0, 0.0, 1.0])),
    )).T
    return R.from_matrix(chest_basis).as_quat()


def build_knee_rotation(hip, knee, ankle, lateral_hint):
    return build_direction_rotation(
        forward=hip - knee,
        up_hint=ankle - knee,
        right_hint=lateral_hint,
    )

def osc_build_bundle(trackers):
    builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    for tracker in trackers:
        builder.add_content(osc_build_msg(tracker['name'], "position", tracker['position']))
        if "rotation" in tracker and tracker["rotation"] is not None:
            builder.add_content(osc_build_msg(tracker['name'], "rotation", tracker['rotation']))
    return builder.build()

class VRChatOSCBackend(Backend):

    def __init__(self, **kwargs):
        self.prev_pose3d = np.zeros((29,3))
        self.tracker_memory = {}
        self.pending_head_position_snap = True
        self.pending_head_rotation_snap = True
        self.last_debug_time = 0.0
        self.frames_without_trackers = 0
        self.send_extended_trackers = False
        self.last_auto_scale_debug_time = 0.0

    def onparamchanged(self, params):
        pass

    def connect(self, params):
        if hasattr(params, "backend_ip") and hasattr(params, "backend_port"):
            self.client = udp_client.UDPClient(params.backend_ip, params.backend_port)
        else:
            self.client = udp_client.UDPClient("127.0.0.1", 9000)
        self.pending_head_position_snap = True
        self.pending_head_rotation_snap = True
        self.tracker_memory = {}
        self.last_debug_time = time.time()
        self.frames_without_trackers = 0
        print(f"INFO: VRChat OSC target set to {params.backend_ip}:{params.backend_port}")

    def _smooth_tracker(self, name, position, rotation_deg, blend):
        if name not in self.tracker_memory:
            self.tracker_memory[name] = {
                "position": np.array(position, dtype=float),
                "rotation": np.array(rotation_deg, dtype=float),
            }
            return self.tracker_memory[name]["position"], self.tracker_memory[name]["rotation"]

        memory = self.tracker_memory[name]
        memory["position"] = memory["position"] * blend + np.array(position, dtype=float) * (1.0 - blend)
        delta = np.array(rotation_deg, dtype=float) - memory["rotation"]
        delta = (delta + 180.0) % 360.0 - 180.0
        memory["rotation"] = memory["rotation"] + delta * (1.0 - blend)
        return memory["position"], memory["rotation"]

    def _queue_tracker(self, trackers, name, position, rotation_deg, visibility, smoothing):
        if visibility < 0.42:
            return
        position, rotation_deg = self._smooth_tracker(name, position, rotation_deg, smoothing)
        trackers.append({
            "name": str(name),
            "position": position,
            "rotation": rotation_deg,
        })

    def _build_chest_rotation(self, pose3d):
        return build_chest_rotation(pose3d)

    def _build_knee_rotation(self, hip, knee, ankle, lateral_hint):
        return build_knee_rotation(hip, knee, ankle, lateral_hint)

    def _safe_send(self, message, label=None):
        try:
            self.client.send(message)
            return True
        except (BlockingIOError, OSError) as exc:
            if label is not None:
                print(f"WARNING: VRChat OSC send skipped for {label}: {exc}")
            else:
                print(f"WARNING: VRChat OSC send skipped: {exc}")
            return False

    def _get_vrchat_pose_scale(self, params, pose3d):
        pose_scale = float(params.posescale)

        if params.calib_scale and pose_scale <= 1.01:
            skeleton_bounds = np.max(pose3d, axis=0) - np.min(pose3d, axis=0)
            skeleton_height = float(skeleton_bounds[1]) if skeleton_bounds[1] > 1e-6 else 0.0
            if skeleton_height > 0.0:
                pose_scale = float(params.osc_target_height / skeleton_height)
                now = time.time()
                if now - self.last_auto_scale_debug_time > 2.0:
                    self.last_auto_scale_debug_time = now
                    print(
                        f"INFO: VRChat OSC runtime auto-scale using {pose_scale:.3f} "
                        f"from target height {params.osc_target_height:.2f}m"
                    )

        return pose_scale

    def _maybe_send_head_alignment(self, params, transformed_pose, pose3d, visibility):
        head_vis = visibility_average(visibility, (7, 8, 9))
        if head_vis < 0.55:
            return

        head_vec = transformed_pose[8] - transformed_pose[7]
        if np.linalg.norm(head_vec) < 1e-6:
            head_vec = np.array([0.0, 0.12, 0.02])
        head_anchor = transformed_pose[7] + head_vec * 0.65

        shoulder_axis = pose3d[13] - pose3d[12]
        torso_axis = pose3d[7] - pose3d[6]
        head_rot = build_direction_rotation(
            forward=torso_axis,
            up_hint=head_vec,
            right_hint=shoulder_axis,
        )

        if self.pending_head_position_snap or params.osc_realign_now:
            if self._safe_send(osc_build_msg("head", "position", pose_to_vrchat_position(head_anchor)), "head-position"):
                self.pending_head_position_snap = False

        if self.pending_head_rotation_snap or params.osc_realign_now:
            if self._safe_send(osc_build_msg("head", "rotation", quat_to_vrchat_euler(head_rot)), "head-rotation"):
                self.pending_head_rotation_snap = False

        params.osc_realign_now = False

    def updatepose(self, params, pose3d, rots, hand_rots, visibility=None):
    
        #pose3d[:,1] = -pose3d[:,1]      #flip the positions as coordinate system is different from steamvr
        #pose3d[:,0] = -pose3d[:,0]
        
        pose3d = self.prev_pose3d*params.additional_smoothing + pose3d*(1-params.additional_smoothing)
        self.prev_pose3d = pose3d
    
        headsetpos = [float(0),float(0),float(0)]
        headsetrot = R.from_quat([float(0),float(0),float(0),float(1)])

        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.
        if params.recalibrate:
            print("frame to recalibrate")
        else:
            pose_scale = self._get_vrchat_pose_scale(params, pose3d)
            pose3d = pose3d * pose_scale     #rescale skeleton to calibrated height
            #print(pose3d)
            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            transformed_pose = pose3d - offset
            self._maybe_send_head_alignment(params, transformed_pose, pose3d, visibility)
            if not params.preview_skeleton:
                trackers = []
                base_smoothing = float(np.clip(params.additional_smoothing * 0.6, 0.0, 0.8))
                if not params.ignore_hip:
                    self._queue_tracker(
                        trackers,
                        1,
                        pose_to_vrchat_position(transformed_pose[6]),
                        quat_to_vrchat_euler(rots[0]),
                        visibility_average(visibility, (2, 3, 6, 7)),
                        base_smoothing,
                    )
                else:
                    pass

                self._queue_tracker(
                    trackers,
                    2,
                    pose_to_vrchat_position(transformed_pose[0]),
                    quat_to_vrchat_euler(rots[1]),
                    visibility_average(visibility, (0, 1, 17, 18, 19)),
                    base_smoothing,
                )
                self._queue_tracker(
                    trackers,
                    3,
                    pose_to_vrchat_position(transformed_pose[5]),
                    quat_to_vrchat_euler(rots[2]),
                    visibility_average(visibility, (4, 5, 20, 21, 22)),
                    base_smoothing,
                )

                if self.send_extended_trackers:
                    chest_rotation = self._build_chest_rotation(pose3d)
                    self._queue_tracker(
                        trackers,
                        4,
                        pose_to_vrchat_position(transformed_pose[7]),
                        quat_to_vrchat_euler(chest_rotation),
                        visibility_average(visibility, (7, 12, 13)),
                        min(0.9, base_smoothing + 0.08),
                    )

                    hip_span = pose3d[3] - pose3d[2]
                    right_knee_rot = self._build_knee_rotation(pose3d[3], pose3d[4], pose3d[5], hip_span)
                    left_knee_rot = self._build_knee_rotation(pose3d[2], pose3d[1], pose3d[0], -hip_span)
                    self._queue_tracker(
                        trackers,
                        5,
                        pose_to_vrchat_position(transformed_pose[4]),
                        quat_to_vrchat_euler(right_knee_rot),
                        visibility_average(visibility, (3, 4, 5)),
                        min(0.92, base_smoothing + 0.12),
                    )
                    self._queue_tracker(
                        trackers,
                        6,
                        pose_to_vrchat_position(transformed_pose[1]),
                        quat_to_vrchat_euler(left_knee_rot),
                        visibility_average(visibility, (0, 1, 2)),
                        min(0.92, base_smoothing + 0.12),
                    )

                if params.use_hands:
                    # Sending hand trackers unsupported
                    pass
                if len(trackers) > 0:
                    for tracker in trackers:
                        self._safe_send(
                            osc_build_msg(tracker["name"], "position", tracker["position"]),
                            f"tracker-{tracker['name']}-position",
                        )
                        if tracker.get("rotation") is not None:
                            self._safe_send(
                                osc_build_msg(tracker["name"], "rotation", tracker["rotation"]),
                                f"tracker-{tracker['name']}-rotation",
                            )
                    self.frames_without_trackers = 0
                else:
                    self.frames_without_trackers += 1

                now = time.time()
                if now - self.last_debug_time > 2.0:
                    self.last_debug_time = now
                    avg_vis = float(np.mean(visibility)) if visibility is not None else 1.0
                    if len(trackers) > 0:
                        tracker_names = ", ".join(str(tracker["name"]) for tracker in trackers)
                        print(
                            f"INFO: Sent {len(trackers)} VRChat OSC trackers to "
                            f"{params.backend_ip}:{params.backend_port} [{tracker_names}]"
                        )
                    else:
                        print(
                            "INFO: No VRChat OSC trackers were sent this frame window. "
                            f"Avg visibility={avg_vis:.2f}, frames_without_trackers={self.frames_without_trackers}"
                        )

            else:
                # Preview skeleton unsupported
                pass
        return True

    def disconnect(self):
        pass
