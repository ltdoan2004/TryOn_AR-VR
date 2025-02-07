import sys
import math
import cv2
import numpy as np
import mediapipe as mp
import pyglet
import trimesh
import pyrender

class RingViewer(pyrender.Viewer):
    def __init__(self, scene, camera_index=0, *args, **kwargs):
        # 1. Mở camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Không mở được camera.")
            sys.exit(1)

        # 2. Tạo MediaPipe Hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # 3. KHỞI TẠO GIÁ TRỊ CHO self.bg_texture TRƯỚC
        # Dù tạm thời là None hoặc 1x1 để tránh lỗi AttributeError nếu on_draw() chạy sớm
        self.bg_texture = None

        # 4. Gọi constructor của pyrender.Viewer
        super().__init__(scene, *args, **kwargs)

        # 5. Bây giờ GL context đã sẵn sàng, ta có thể tạo texture thật
        #    Nếu muốn tạo sẵn 1x1 (dummy):
        self.bg_texture = pyglet.image.Texture.create(1, 1)

        # 6. Schedule update (30 FPS)
        pyglet.clock.schedule_interval(self.update_mediapipe, 1.0 / 30.0)

    def update_mediapipe(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        # ... Xử lý MediaPipe, cập nhật pose nhẫn ...

        # Lưu frame vào bg_texture (nếu đã sẵn sàng)
        if self.bg_texture is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_cam, w_cam, c = frame_rgb.shape
            image_data = pyglet.image.ImageData(
                width=w_cam, height=h_cam, format='RGB',
                data=frame_rgb.tobytes(), pitch=-w_cam*c
            )

            # Nếu kích thước thay đổi, tạo lại texture
            if (self.bg_texture.width != w_cam or
                self.bg_texture.height != h_cam):
                self.bg_texture = pyglet.image.Texture.create(w_cam, h_cam)

            self.bg_texture.blit_into(image_data, 0, 0, 0)

    def on_draw(self):
        """Vẽ background = camera feed, sau đó vẽ scene 3D."""
        w_view, h_view = self.viewport_size

        pyglet.gl.glDisable(pyglet.gl.GL_DEPTH_TEST)
        if self.bg_texture is not None:
            self.bg_texture.blit(0, 0, width=w_view, height=h_view)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)

        super().on_draw()

    def on_close(self):
        self.cap.release()
        self.hands.close()
        super().on_close()

def main():
    ring_glb_path = "data/ring2_webgi.glb"
    mesh = trimesh.load(ring_glb_path, force='mesh')
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.3, 0.3, 0.3]
    )
    scene.add(render_mesh, pose=np.eye(4), name="my_ring_node")

    camera = pyrender.PerspectiveCamera(yfov=np.radians(45.0), znear=0.1, zfar=2000.0)
    camera_pose = np.eye(4)
    camera_pose[2,3] = 1000.0
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    viewer = RingViewer(
        scene,
        camera_index=0,
        use_raymond_lighting=False,
        viewport_size=(640, 480),
        run_in_thread=False
    )
    print("Kết thúc chương trình.")

if __name__ == "__main__":
    main()
