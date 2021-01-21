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
        self.run_in_thread = False
        self.scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3, 1.0]
        )
        if camera is None:
            camera = pyrender.OrthographicCamera(ymag=1, xmag=1)

        if camera_pose is None:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = np.array([0, 0, -2])
            camera_pose[0, 0] *= -1.0

        self.groups = {
            "body": [],
            "keypoints": []
        }

        self.scene.add(camera, pose=camera_pose)

    def start(self,
              use_reymond_lighting=True,
              run_in_thread=True):

        self.run_in_thread = run_in_thread
        self.viewer = pyrender.Viewer(
            self.scene,
            run_in_thread=run_in_thread,
            use_reymond_lighting=use_reymond_lighting
        )

    def stop(self):
        self.viewer.close_external()
        while self.viewer.is_active:
            pass

    def requires_lock(self):
        return self.run_in_thread and self.viewer

    def release(self):
        if self.requires_lock():
            self.viewer.render_lock.release()

    def acquire(self):
        if self.requires_lock():
            self.viewer.render_lock.acquire()

    def remove_from_group(self, group_name, name):
        group = self.groups[group_name]
        if group is None:
            return
        self.groups[group_name] = [node for node in group if node.name != name]

    def add_to_group(self, group_name, node):
        # store node in a group for easier access later
        if group_name is not None:
            cur_group = self.groups[group_name]
            if cur_group is None:
                self.groups[group_name] = []
                cur_group = self.groups[group_name]
            cur_group.append(node)

    def render_points(self, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0], name=None, group_name=None):

        self.acquire()
        node = render_points(self.scene, points=points,
                             radius=radius, color=color, name=name)
        self.release()

        self.add_to_group(group_name, node)

        return node

    def render_keypoints(self, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0]):
        """Utility method to render joints, executes render_points with a fixed name

        Args:
            points ([type]): [description]
            radius (float, optional): [description]. Defaults to 0.005.
            color (list, optional): [description]. Defaults to [0.0, 0.0, 1.0, 1.0].
        """
        self.remove_from_group("keypoints", "ops_keypoints")

        return self.render_points(points, radius, color=color, name="ops_keypoints", group_name="keypoints")

    def render_joints(self, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0]):
        """Utility method to render joints, executes render_points with a fixed name

        Args:
            points ([type]): [description]
            radius (float, optional): [description]. Defaults to 0.005.
            color (list, optional): [description]. Defaults to [0.0, 0.0, 1.0, 1.0].
        """
        self.remove_from_group("body", "body_joints")

        return self.render_points(points, radius, color=color, name="body_joints", group_name="body")

    def render_model(
            self,
            model,
            model_out,
            color=[1.0, 0.3, 0.3, 0.8],
            replace=True
    ):
        if model_out is None:
            model_out = model()

        self.remove_from_group("body", "body_mesh")

        self.acquire()
        node = render_model(self.scene, model, model_out,
                            color, "body_mesh", replace=replace)
        self.release()

        self.add_to_group("body", node)
        return node

    def set_group_transform(self, group_name, rotation, translation):
        # create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation

        self.set_group_pose(group_name, pose)

    def set_group_pose(self, group_name, pose):
        group = self.groups[group_name]
        if group is None:
            print("[render] group with name does not exist:", group_name)
            return
        self.acquire()

        for node in group:
            self.scene.set_pose(node, pose)

        self.release()

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
