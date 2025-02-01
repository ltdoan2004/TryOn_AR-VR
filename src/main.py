import os
# Trên macOS, hãy sử dụng nền tảng macosx thay vì osmesa
os.environ["PYOPENGL_PLATFORM"] = "macosx"

import cv2
import mediapipe as mp
import numpy as np
import pyrender
import trimesh
import math

# -------------------------
# 1. CÀI ĐẶT THIẾT LẬP 3D RENDER
# -------------------------
# Đường dẫn file .glb của nhẫn
ring_glb_path = "data/ring2_webgi.glb"  # Thay đổi đường dẫn cho phù hợp

# Load file .glb bằng trimesh (chú ý: nếu file là scene có nhiều mesh, có thể cần xử lý khác)
try:
    # force='mesh' => chuyển về dạng mesh nếu file là một scene
    mesh = trimesh.load(ring_glb_path, force='mesh')
except Exception as e:
    print("Lỗi khi load file .glb:", e)
    exit()

# Tạo đối tượng mesh cho Pyrender
render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

# Tạo scene của pyrender với background trong suốt, ánh sáng ambient
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
# Thêm model nhẫn vào scene (mặc định pose = ma trận đơn vị)
ring_node = scene.add(render_mesh, pose=np.eye(4))

# -------------------------
# 2. CÀI ĐẶT CAMERA OFFSCREEN (ORTHOGRAPHIC)
# -------------------------
# Camera chiếu trực giao để map không gian 3D -> pixel.
frame_width, frame_height = 640, 480
xmag = frame_width / 2.0  # bán kính chiếu theo chiều X
ymag = frame_height / 2.0  # bán kính chiếu theo chiều Y

camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.1, zfar=1000.0)

# Đặt camera tại (0, 0, 500) nhìn về -z
camera_pose = np.eye(4)
camera_pose[2, 3] = 500
scene.add(camera, pose=camera_pose)

# Tạo renderer offscreen
renderer = pyrender.OffscreenRenderer(viewport_width=frame_width,
                                      viewport_height=frame_height,
                                      point_size=1.0)

# -------------------------
# 3. CÀI ĐẶT MEDIAPIPE HANDS
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------------
# 4. MỞ CAMERA (OpenCV)
# -------------------------
cap = cv2.VideoCapture(0)  # Thường là 0 cho cam mặc định
if not cap.isOpened():
    print("Không mở được camera.")
    exit()

print("Nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ảnh (mirror)
    frame = cv2.flip(frame, 1)
    orig_frame = frame.copy()

    # Chuyển sang RGB để MediaPipe xử lý
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Biến chứa overlay (ảnh RGBA) của nhẫn (sau khi render)
    ring_overlay = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Vẽ landmark bàn tay lên frame để debug
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            # Landmark 16: đầu ngón áp út
            lm_ring = handLms.landmark[16]
            x_ring = int(lm_ring.x * w)
            y_ring = int(lm_ring.y * h)

            # Landmark 14: đốt thứ hai của ngón áp út (gần bàn tay hơn)
            lm_base = handLms.landmark[14]
            x_base = int(lm_base.x * w)
            y_base = int(lm_base.y * h)

            # Tính góc xoay ngón
            angle = math.degrees(math.atan2(y_ring - y_base, x_ring - x_base))

            # Chuyển đổi từ tọa độ pixel sang "world" (camera orthographic)
            world_x = x_ring - (w / 2.0)
            world_y = (h / 2.0) - y_ring
            ring_z = 0.0

            # Tạo ma trận transform (4x4)
            transform = np.eye(4)
            transform[0, 3] = world_x
            transform[1, 3] = world_y
            transform[2, 3] = ring_z

            # Xoay quanh trục Z
            theta = np.radians(-angle)
            rot_z = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta),  np.cos(theta), 0, 0],
                [0,              0,             1, 0],
                [0,              0,             0, 1]
            ])
            transform = transform @ rot_z

            # Scale model (điều chỉnh kích thước nếu cần)
            scale_factor = 1.0
            scale_matrix = np.diag([scale_factor, scale_factor, scale_factor, 1])
            transform = transform @ scale_matrix

            # Đặt pose cho node nhẫn
            scene.set_pose(ring_node, pose=transform)

            # Render scene (Pyrender -> ảnh RGBA)
            try:
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
                ring_overlay = color
            except Exception as render_error:
                print("Lỗi khi render scene:", render_error)
            break  # Xử lý 1 tay đầu tiên

    # -------------------------
    # 5. PHỐI (COMPOSITE) ẢNH RENDER VÀO FRAME
    # -------------------------
    if ring_overlay is not None:
        # Nếu ảnh render chưa có kênh alpha, thêm vào
        if ring_overlay.shape[2] == 3:
            alpha_channel = np.ones((ring_overlay.shape[0], ring_overlay.shape[1], 1),
                                    dtype=ring_overlay.dtype) * 255
            ring_overlay = np.concatenate((ring_overlay, alpha_channel), axis=2)

        # Tách alpha, chuẩn hoá về [0,1]
        alpha = ring_overlay[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]

        # Chuyển đổi sang float để tính toán composite
        frame_float = orig_frame.astype(float)
        overlay_float = ring_overlay[:, :, :3].astype(float)

        # Công thức: composite = overlay * alpha + frame * (1 - alpha)
        composite = frame_float * (1 - alpha) + overlay_float * alpha
        frame = composite.astype(np.uint8)

    # Hiển thị kết quả
    cv2.imshow("3D Ring Try-On", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC
        break

# Dọn dẹp
cap.release()
renderer.delete()
cv2.destroyAllWindows()
