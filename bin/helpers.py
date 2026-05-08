import time

import socket
import cv2

import numpy as np
from sys import platform
from scipy.spatial.transform import Rotation as R
import cv2
import threading
import sys


POSE_BONE_PAIRS = (
    (2, 1), (1, 0),
    (3, 4), (4, 5),
    (2, 3),
    (7, 2), (7, 3),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15),
)


def safe_normalize(vec, fallback=None):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return fallback / fallback_norm
    return vec / norm


def solve_knee_position(hip, ankle, upper_leg, lower_leg, bend_hint, fallback_knee=None):
    hip = np.asarray(hip, dtype=float)
    ankle = np.asarray(ankle, dtype=float)
    bend_hint = np.asarray(bend_hint, dtype=float)

    hip_to_ankle = ankle - hip
    distance = np.linalg.norm(hip_to_ankle)
    if distance < 1e-6:
        direction = np.array([0.0, -1.0, 0.0], dtype=float)
        bend_plane = safe_normalize(bend_hint, fallback=np.array([0.0, 0.0, 1.0], dtype=float))
        return hip + direction * upper_leg + bend_plane * min(upper_leg, lower_leg) * 0.15

    max_reach = max(upper_leg + lower_leg - 1e-6, 1e-6)
    min_reach = max(abs(upper_leg - lower_leg) + 1e-6, 1e-6)
    distance = float(np.clip(distance, min_reach, max_reach))
    direction = hip_to_ankle / np.linalg.norm(hip_to_ankle)

    projected = bend_hint - direction * np.dot(bend_hint, direction)
    if np.linalg.norm(projected) < 1e-6 and fallback_knee is not None:
        fallback_dir = np.asarray(fallback_knee, dtype=float) - hip
        projected = fallback_dir - direction * np.dot(fallback_dir, direction)
    plane_normal = safe_normalize(projected, fallback=np.array([0.0, 0.0, 1.0], dtype=float))

    hip_component = (upper_leg ** 2 - lower_leg ** 2 + distance ** 2) / (2.0 * distance)
    knee_height = max(upper_leg ** 2 - hip_component ** 2, 0.0) ** 0.5
    return hip + direction * hip_component + plane_normal * knee_height

def draw_pose(frame,pose,size):
    pose = pose*size
    for sk in EDGES:
        cv2.line(frame,(int(pose[sk[0],1]),int(pose[sk[0],0])),(int(pose[sk[1],1]),int(pose[sk[1],0])),(0,255,0),3)

def mediapipeTo3dpose(lms):
    #33 pose landmarks as in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    #convert landmarks returned by mediapipe to skeleton that I use.
    #lms = results.pose_world_landmarks.landmark
    
    pose = np.zeros((29,3))

    pose[0]=[lms[28].x,lms[28].y,lms[28].z]
    pose[1]=[lms[26].x,lms[26].y,lms[26].z]
    pose[2]=[lms[24].x,lms[24].y,lms[24].z]
    pose[3]=[lms[23].x,lms[23].y,lms[23].z]
    pose[4]=[lms[25].x,lms[25].y,lms[25].z]
    pose[5]=[lms[27].x,lms[27].y,lms[27].z]

    pose[6]=[0,0,0]

    #some keypoints in mediapipe are missing, so we calculate them as avarage of two keypoints
    pose[7]=[lms[12].x/2+lms[11].x/2,lms[12].y/2+lms[11].y/2,lms[12].z/2+lms[11].z/2]
    pose[8]=[lms[10].x/2+lms[9].x/2,lms[10].y/2+lms[9].y/2,lms[10].z/2+lms[9].z/2]

    pose[9]=[lms[0].x,lms[0].y,lms[0].z]

    pose[10]=[lms[15].x,lms[15].y,lms[15].z]
    pose[11]=[lms[13].x,lms[13].y,lms[13].z]
    pose[12]=[lms[11].x,lms[11].y,lms[11].z]

    pose[13]=[lms[12].x,lms[12].y,lms[12].z]
    pose[14]=[lms[14].x,lms[14].y,lms[14].z]
    pose[15]=[lms[16].x,lms[16].y,lms[16].z]

    pose[16]=[pose[6][0]/2+pose[7][0]/2,pose[6][1]/2+pose[7][1]/2,pose[6][2]/2+pose[7][2]/2]

    #right foot
    pose[17] = [lms[31].x,lms[31].y,lms[31].z]  #forward
    pose[18] = [lms[29].x,lms[29].y,lms[29].z]  #back  
    pose[19] = [lms[25].x,lms[25].y,lms[25].z]  #up
    
    #left foot
    pose[20] = [lms[32].x,lms[32].y,lms[32].z]  #forward
    pose[21] = [lms[30].x,lms[30].y,lms[30].z]  #back
    pose[22] = [lms[26].x,lms[26].y,lms[26].z]  #up
    
    #right hand
    pose[23] = [lms[17].x,lms[17].y,lms[17].z]  #forward
    pose[24] = [lms[15].x,lms[15].y,lms[15].z]  #back
    pose[25] = [lms[19].x,lms[19].y,lms[19].z]  #up
    
    #left hand
    pose[26] = [lms[18].x,lms[18].y,lms[18].z]  #forward
    pose[27] = [lms[16].x,lms[16].y,lms[16].z]  #back
    pose[28] = [lms[20].x,lms[20].y,lms[20].z]  #up

    return pose


def mediapipeToVisibility(lms):
    visibility = np.ones(29, dtype=float)

    def vis(idx):
        return float(getattr(lms[idx], "visibility", 1.0))

    visibility[0] = vis(28)
    visibility[1] = vis(26)
    visibility[2] = vis(24)
    visibility[3] = vis(23)
    visibility[4] = vis(25)
    visibility[5] = vis(27)
    visibility[6] = (visibility[2] + visibility[3]) / 2
    visibility[7] = (vis(12) + vis(11)) / 2
    visibility[8] = (vis(10) + vis(9)) / 2
    visibility[9] = vis(0)
    visibility[10] = vis(15)
    visibility[11] = vis(13)
    visibility[12] = vis(11)
    visibility[13] = vis(12)
    visibility[14] = vis(14)
    visibility[15] = vis(16)
    visibility[16] = (visibility[6] + visibility[7]) / 2
    visibility[17] = vis(31)
    visibility[18] = vis(29)
    visibility[19] = vis(25)
    visibility[20] = vis(32)
    visibility[21] = vis(30)
    visibility[22] = vis(26)
    visibility[23] = vis(17)
    visibility[24] = vis(15)
    visibility[25] = vis(19)
    visibility[26] = vis(18)
    visibility[27] = vis(16)
    visibility[28] = vis(20)
    return np.clip(visibility, 0.0, 1.0)


class PoseStabilizer:
    def __init__(self):
        self.prev_pose = None
        self.prev_velocity = None
        self.bone_lengths = None
        self.floor_height = None
        self.locked_feet = {
            0: {"position": None, "frames": 0},
            5: {"position": None, "frames": 0},
        }

    def reset(self):
        self.prev_pose = None
        self.prev_velocity = None
        self.bone_lengths = None
        self.floor_height = None
        for foot in self.locked_feet.values():
            foot["position"] = None
            foot["frames"] = 0

    def _ensure_bone_lengths(self, pose3d):
        if self.bone_lengths is None:
            self.bone_lengths = {}
        for start, end in POSE_BONE_PAIRS:
            if (start, end) not in self.bone_lengths:
                dist = np.linalg.norm(pose3d[end] - pose3d[start])
                if dist > 1e-6:
                    self.bone_lengths[(start, end)] = dist

    def _apply_temporal_filter(self, pose3d, visibility, dt, params):
        if self.prev_pose is None:
            self.prev_pose = pose3d.copy()
            self.prev_velocity = np.zeros_like(pose3d)
            return pose3d.copy()

        predicted_pose = self.prev_pose + self.prev_velocity * dt * params.stabilizer_prediction
        filtered_pose = pose3d.copy()

        for joint_id in range(pose3d.shape[0]):
            speed = np.linalg.norm((pose3d[joint_id] - self.prev_pose[joint_id]) / dt)
            alpha = params.stabilizer_amount
            alpha += (1.0 - visibility[joint_id]) * params.stabilizer_visibility_bias
            alpha -= min(speed, 2.0) * params.stabilizer_motion_response
            alpha = float(np.clip(alpha, params.stabilizer_min_blend, params.stabilizer_max_blend))
            filtered_pose[joint_id] = predicted_pose[joint_id] * alpha + pose3d[joint_id] * (1.0 - alpha)

        self.prev_velocity = (filtered_pose - self.prev_pose) / dt
        self.prev_pose = filtered_pose.copy()
        return filtered_pose

    def _stabilize_pelvis_and_torso(self, pose3d):
        hip_center = (pose3d[2] + pose3d[3]) / 2
        shoulder_center = pose3d[7]

        hip_dir = safe_normalize(pose3d[3] - pose3d[2], fallback=np.array([1.0, 0.0, 0.0]))
        hip_half_width = self.bone_lengths.get((2, 3), np.linalg.norm(pose3d[3] - pose3d[2])) / 2
        pose3d[2] = hip_center - hip_dir * hip_half_width
        pose3d[3] = hip_center + hip_dir * hip_half_width

        for start, end in ((7, 2), (7, 3), (7, 12), (7, 13)):
            target_length = self.bone_lengths.get((start, end))
            if target_length is None:
                continue
            direction = safe_normalize(pose3d[end] - pose3d[start], fallback=pose3d[end] - shoulder_center)
            pose3d[end] = pose3d[start] + direction * target_length

    def _apply_torso_lean(self, pose3d, params):
        torso_vec = pose3d[7] - pose3d[6]
        torso_horizontal = torso_vec.copy()
        torso_horizontal[1] = 0.0
        horizontal_len = np.linalg.norm(torso_horizontal)
        if horizontal_len < 1e-5:
            return pose3d

        extra_lean = torso_horizontal * params.torso_lean_strength
        # Shift the lower body opposite the shoulder lean so bends read more clearly in-avatar.
        for joint_id in (0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22):
            pose3d[joint_id] -= extra_lean
        return pose3d

    def _enforce_bone_lengths(self, pose3d):
        self._stabilize_pelvis_and_torso(pose3d)
        for _ in range(2):
            for start, end in POSE_BONE_PAIRS:
                target_length = self.bone_lengths.get((start, end))
                if target_length is None:
                    continue
                direction = safe_normalize(pose3d[end] - pose3d[start], fallback=pose3d[end] - pose3d[start])
                pose3d[end] = pose3d[start] + direction * target_length
        pose3d[6] = (pose3d[2] + pose3d[3]) / 2
        pose3d[16] = (pose3d[6] + pose3d[7]) / 2
        pose3d[8] = (pose3d[9] + pose3d[7]) / 2
        return pose3d

    def _update_floor_height(self, pose3d, visibility, params):
        candidates = []
        for joint_id in (0, 5, 17, 18, 20, 21):
            if visibility[joint_id] >= params.stabilizer_floor_visibility:
                candidates.append(float(pose3d[joint_id][1]))
        if not candidates:
            return

        current_floor = min(candidates)
        if self.floor_height is None:
            self.floor_height = current_floor
        else:
            self.floor_height = (
                self.floor_height * params.stabilizer_floor_smoothing
                + current_floor * (1.0 - params.stabilizer_floor_smoothing)
            )

    def _apply_foot_lock(self, pose3d, visibility, params):
        if self.floor_height is None:
            return pose3d

        for ankle_idx, knee_idx in ((0, 1), (5, 4)):
            foot_state = self.locked_feet[ankle_idx]
            foot_speed = np.linalg.norm(self.prev_velocity[ankle_idx]) if self.prev_velocity is not None else 0.0
            foot_near_floor = abs(pose3d[ankle_idx][1] - self.floor_height) < params.foot_lock_height_threshold
            foot_visible = visibility[ankle_idx] >= params.foot_lock_visibility_threshold

            if foot_visible and foot_near_floor and foot_speed < params.foot_lock_speed_threshold:
                foot_state["frames"] += 1
                if foot_state["position"] is None:
                    foot_state["position"] = pose3d[ankle_idx].copy()
                    foot_state["position"][1] = self.floor_height
                else:
                    foot_state["position"] = (
                        foot_state["position"] * params.foot_lock_anchor_blend
                        + pose3d[ankle_idx] * (1.0 - params.foot_lock_anchor_blend)
                    )
                    foot_state["position"][1] = self.floor_height
            else:
                foot_state["frames"] = max(foot_state["frames"] - 1, 0)
                if foot_state["frames"] == 0:
                    foot_state["position"] = None

            if foot_state["position"] is not None and foot_state["frames"] >= params.foot_lock_frames:
                locked_position = foot_state["position"].copy()
                locked_position[1] = self.floor_height
                pose3d[ankle_idx] = (
                    locked_position * params.foot_lock_amount
                    + pose3d[ankle_idx] * (1.0 - params.foot_lock_amount)
                )
                hip_idx = 2 if ankle_idx == 0 else 3
                upper_leg = self.bone_lengths.get((hip_idx, knee_idx))
                lower_leg = self.bone_lengths.get((knee_idx, ankle_idx))
                if upper_leg is not None and lower_leg is not None:
                    hip_span = pose3d[3] - pose3d[2]
                    bend_hint = np.cross(
                        hip_span if ankle_idx == 0 else -hip_span,
                        pose3d[ankle_idx] - pose3d[hip_idx],
                    )
                    pose3d[knee_idx] = solve_knee_position(
                        pose3d[hip_idx],
                        pose3d[ankle_idx],
                        upper_leg,
                        lower_leg,
                        bend_hint,
                        fallback_knee=pose3d[knee_idx],
                    )

        return pose3d

    def update(self, pose3d, visibility, dt, params):
        dt = float(np.clip(dt, 1.0 / 240.0, 0.1))
        visibility = np.asarray(visibility, dtype=float)

        self._ensure_bone_lengths(pose3d)
        pose3d = self._apply_temporal_filter(pose3d, visibility, dt, params)
        pose3d = self._apply_torso_lean(pose3d, params)
        pose3d = self._enforce_bone_lengths(pose3d)
        self._update_floor_height(pose3d, visibility, params)
        pose3d = self._apply_foot_lock(pose3d, visibility, params)
        pose3d = self._enforce_bone_lengths(pose3d)

        self.prev_pose = pose3d.copy()
        return pose3d

def keypoints_to_original(scale,center,points):
    scores = points[:,2]
    points -= 0.5
    points *= scale
    points[:,0] += center[0]
    points[:,1] += center[1]
    
    points[:,2] = scores
    
    return points

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def get_rot_hands(pose3d):

    hand_r_f = pose3d[26]
    hand_r_b = pose3d[27]
    hand_r_u = pose3d[28]
    
    hand_l_f = pose3d[23]
    hand_l_b = pose3d[24]
    hand_l_u = pose3d[25]
    
    # left hand
    
    x = hand_l_f - hand_l_b
    w = hand_l_u - hand_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_hand_rot = np.vstack((z, y, -x)).T
    
    # right hand
    
    x = hand_r_f - hand_r_b
    w = hand_r_u - hand_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_hand_rot = np.vstack((z, y, -x)).T

    r_hand_rot = R.from_matrix(r_hand_rot).as_quat()
    l_hand_rot = R.from_matrix(l_hand_rot).as_quat()
    
    return l_hand_rot, r_hand_rot

def get_rot_mediapipe(pose3d):
    hip_left = pose3d[2]
    hip_right = pose3d[3]
    hip_up = pose3d[16]
    
    foot_l_f = pose3d[20]
    foot_l_b = pose3d[21]
    foot_l_u = pose3d[22]
    
    foot_r_f = pose3d[17]
    foot_r_b = pose3d[18]
    foot_r_u = pose3d[19]
    
    # hip
    
    x = hip_right - hip_left
    w = hip_up - hip_left
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T
    
    # left foot
    
    x = foot_l_f - foot_l_b
    w = foot_l_u - foot_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_foot_rot = np.vstack((x, y, z)).T
    
    # right foot
    
    x = foot_r_f - foot_r_b
    w = foot_r_u - foot_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_foot_rot = np.vstack((x, y, z)).T
    
    hip_rot = R.from_matrix(hip_rot).as_quat()
    r_foot_rot = R.from_matrix(r_foot_rot).as_quat()
    l_foot_rot = R.from_matrix(l_foot_rot).as_quat()
    
    return hip_rot, l_foot_rot, r_foot_rot

    
def get_rot(pose3d):

    ## guesses
    hip_left = 2
    hip_right = 3
    hip_up = 16
    
    knee_left = 1
    knee_right = 4
    
    ankle_left = 0
    ankle_right = 5
    
    # hip
    
    x = pose3d[hip_right] - pose3d[hip_left]
    w = pose3d[hip_up] - pose3d[hip_left]
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T

    # right leg
    
    y = pose3d[knee_right] - pose3d[ankle_right]
    w = pose3d[hip_right] - pose3d[ankle_right]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_left] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_r_rot = np.vstack((x, y, z)).T

    # left leg
    
    y = pose3d[knee_left] - pose3d[ankle_left]
    w = pose3d[hip_left] - pose3d[ankle_left]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_right] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_l_rot = np.vstack((x, y, z)).T

    rot_hip = R.from_matrix(hip_rot).as_quat()
    rot_leg_r = R.from_matrix(leg_r_rot).as_quat()
    rot_leg_l = R.from_matrix(leg_l_rot).as_quat()
    
    return rot_hip, rot_leg_l, rot_leg_r


def sendToPipe(text):
    if platform.startswith('win32'):
        pipe = open(r'\\.\pipe\ApriltagPipeIn', 'rb+', buffering=0)
        some_data = str.encode(text)
        some_data += b'\0'
        pipe.write(some_data)
        resp = pipe.read(1024)
        pipe.close()
    elif platform.startswith('linux'):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect("/tmp/ApriltagPipeIn")
        some_data = text.encode('utf-8')
        some_data += b'\0'
        client.send(some_data)
        resp = client.recv(1024)
        client.close()
    else:
        print(f"Unsuported platform {sys.platform}")
        raise Exception
    return resp

def sendToSteamVR_(text):
    #Function to send a string to my steamvr driver through a named pipe.
    #open pipe -> send string -> read string -> close pipe
    #sometimes, something along that pipeline fails for no reason, which is why the try catch is needed.
    #returns an array containing the values returned by the driver.
    try:
        resp = sendToPipe(text)
    except:
        return ["error"]

    string = resp.decode("utf-8")
    array = string.split(" ")
    
    return array


def sendToSteamVR(text, num_tries=10, wait_time=0.1):
    # wrapped function sendToSteamVR that detects failed connections
    ret = sendToSteamVR_(text)
    i = 0
    while "error" in ret:
        print("INFO: Error while connecting to SteamVR. Retrying...")
        time.sleep(wait_time)
        ret = sendToSteamVR_(text)
        i += 1
        if i >= num_tries:
            return None # probably better to throw error here and exit the program (assert?)
    
    return ret

    
class CameraStream():
    def __init__(self, params):
        self.params = params
        self.image_ready = False
        # setup camera capture
        if len(params.cameraid) <= 2:
            cameraid = int(params.cameraid)
        else:
            cameraid = params.cameraid
            
        if params.camera_settings: # use advanced settings
            self.cap = cv2.VideoCapture(cameraid, cv2.CAP_DSHOW) 
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        else:
            self.cap = cv2.VideoCapture(cameraid)  

        if not self.cap.isOpened():
            print("ERROR: Could not open camera, try another id/IP")
            shutdown(params)

        if params.camera_height != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(params.camera_height))
            
        if params.camera_width != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(params.camera_width))

        print("INFO: Start camera thread")
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    
    def update(self):
        # continuously grab images
        while True:
            ret, self.image_from_thread = self.cap.read()    
            self.image_ready = True
            
            if ret == 0:
                print("ERROR: Camera capture failed! missed frames.")
                self.params.exit_ready = True
                return
 

def shutdown(params):
    # first save parameters 
    print("INFO: Saving parameters...")
    params.save_params()

    cv2.destroyAllWindows()
    sys.exit("INFO: Exiting... You can close the window after 10 seconds.")
