import numpy as np
from utils import render_model, render_points
import pyrender


class Renderer:
    def __init__(
        self,
        camera=None,
        camera_pose=None
    ) -> None:
        super().__init__()
        self.scene = pyrender.Scene()
        if camera is None:
            camera = pyrender.OrthographicCamera(ymag=1, xmag=1)

        if camera_pose is None:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = np.array([0, 0, -2])
            camera_pose[0, 0] *= -1.0

        self.scene.add(camera, pose=camera_pose)

    def start(self,
              use_reymond_lighting=True,
              run_in_thread=True, **kwargs):

        self.run_in_thread = run_in_thread
        self.viewer = pyrender.Viewer(
            self.scene,
            run_in_thread=run_in_thread,
            use_reymond_lighting=use_reymond_lighting,
            **kwargs
        )

    def stop(self):
        self.viewer.close_external()
        while self.viewer.is_active:
            pass

    def requires_lock(self):
        return self.run_in_thread and self.viewer

    def render_points(self, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0], name=None):
        if self.requires_lock():
            self.viewer.render_lock.acquire()

        node = render_points(self.scene, points=points,
                             radius=radius, colors=colors, name=name)

        if self.requires_lock():
            self.viewer.render_lock.release()

        return node

    def render_keypoints(self, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0]):
        """Utility method to render joints, executes render_points with a fixed name

        Args:
            points ([type]): [description]
            radius (float, optional): [description]. Defaults to 0.005.
            colors (list, optional): [description]. Defaults to [0.0, 0.0, 1.0, 1.0].
        """
        return self.render_points(points, radius, colors, name="ops_keypoints")

    def render_joints(self, points, radius=0.005, colors=[0.0, 0.0, 1.0, 1.0]):
        """Utility method to render joints, executes render_points with a fixed name

        Args:
            points ([type]): [description]
            radius (float, optional): [description]. Defaults to 0.005.
            colors (list, optional): [description]. Defaults to [0.0, 0.0, 1.0, 1.0].
        """
        return self.render_points(points, radius, colors, name="body_joints")

    def render_model(
            self,
            model,
            model_out,
            color=[0.3, 0.3, 0.3, 0.8],
            replace=True
    ):
        if model_out is None:
            model_out = model()

        return render_model(self.scene, model, model_out,
                            color, "body_mesh", replace=replace)

    def set_pose(self, name, pose):
        # find node
        cur_node = None

        for node in self.scene.get_nodes(name):
            if node is not None:
                cur_node = node
                break

        if cur_node is None:
            print("[render] node not found with name", name)
            return

        if self.requires_lock():
            self.viewer.render_lock.acquire()

        self.scene.set_pose(cur_node, pose)

        if self.requires_lock():
            self.viewer.render_lock.release()
