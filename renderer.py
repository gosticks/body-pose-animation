import cv2
import numpy as np
from pyrender import scene
from smplx import SMPLLayer
from smplx.body_models import SMPL
from utils.render import render_model, render_model_with_tfs, render_points, render_camera, render_image_plane
import pyrender


class Renderer:
    def __init__(
        self,
        camera=None,
        camera_pose=None,
        width=1920,
        height=1080,
        light_color=[0.3, 0.3, 0.3, 1.0]
    ) -> None:
        super().__init__()
        self.run_in_thread = False
        self.scene = pyrender.Scene(
            ambient_light=light_color
        )
        self.width = width
        self.height = height

        if camera is None:
            camera = pyrender.OrthographicCamera(ymag=1, xmag=1)

        if camera_pose is None:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = np.array(
                [width / height - 1, 0, width / height])

        self.groups = {
            "body": [],
            "keypoints": []
        }

        self.scene.add(pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=10.0))
        self.scene.add(camera, pose=camera_pose)

    def start(self,
              use_reymond_lighting=True,
              run_in_thread=True,
              viewport_size=None):

        if viewport_size is None:
            viewport_size = (self.width, self.height)

        self.run_in_thread = run_in_thread
        self.viewer = pyrender.Viewer(
            self.scene,
            run_in_thread=run_in_thread,
            use_reymond_lighting=use_reymond_lighting,
            viewport_size=tuple(d // 2 for d in viewport_size)
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

    def remove_from_group(self, group_name, name, remove_node=True):
        group = self.groups[group_name]
        if group is None:
            return

        if remove_node:
            self.acquire()
            nodes = self.scene.get_nodes(name=name)
            if len(nodes) > 0:
                node = nodes.pop()
                if node is not None:
                    self.scene.remove_node(node)
            self.release()
        # filter group array
        self.groups[group_name] = [node for node in group if node.name != name]

    def add_to_group(self, group_name, node):
        # store node in a group for easier access later
        if group_name is not None:
            cur_group = self.groups[group_name]
            if cur_group is None:
                self.groups[group_name] = []
                cur_group = self.groups[group_name]
            cur_group.append(node)

    def render_points(self, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0], name=None, group_name=None, transforms=None):

        self.acquire()
        node = render_points(self.scene, points=points,
                             radius=radius, color=color, name=name, transform=transforms)
        self.release()

        self.add_to_group(group_name, node)

        return node

    def render_camera(self, radius=0.1, height=0.1, color=[0.0, 0.0, 1.0, 1.0], name=None, group_name=None):

        self.acquire()
        node = render_camera(self.scene, radius=radius,
                             height=height, color=color, name=name)
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

    def render_joints(self, points, radius=0.005, color=[0.0, 0.0, 1.0, 1.0], transforms=None):
        """Utility method to render joints, executes render_points with a fixed name

        Args:
            points ([type]): [description]
            radius (float, optional): [description]. Defaults to 0.005.
            color (list, optional): [description]. Defaults to [0.0, 0.0, 1.0, 1.0].
            :param transforms:
        """
        self.remove_from_group("body", "body_joints")

        return self.render_points(points, radius, color=color, name="body_joints", group_name="body",  transforms=transforms)

    def render_model(
            self,
            model: SMPLLayer,
            model_out,
            color=[1.0, 0.3, 0.3, 0.8],
            replace=True,
            keep_pose=True,
            render_joints=True
    ):
        if model_out is None:
            model_out = model()

        if keep_pose:
            node = self.get_node("body_mesh")
            # if node is not None:
            #original_pose = node.pose

        if render_joints:
            self.render_joints(
                model_out.joints.detach().cpu().numpy().squeeze())

        self.remove_from_group("body", "body_mesh")

        self.acquire()
        node = render_model(self.scene, model, model_out,
                            color, "body_mesh", replace=replace)
        self.release()

        self.add_to_group("body", node)
        return node

    def render_model_with_tfs(
            self,
            model: SMPLLayer,
            model_out: SMPL,
            color=[1.0, 0.3, 0.3, 0.8],
            replace=True,
            keep_pose=True,
            render_joints=True,
            transforms=None
    ):
        if model_out is None:
            model_out = model()

        if keep_pose:
            node = self.get_node("body_mesh")
            # if node is not None:
            #original_pose = node.pose

        self.render_joints(model_out.joints.detach(
        ).cpu().numpy().squeeze(), transforms=transforms)

        self.remove_from_group("body", "body_mesh")

        self.acquire()
        node = render_model_with_tfs(self.scene, model, model_out,
                                     color, "body_mesh", replace=replace, transforms=transforms)
        self.release()

        self.add_to_group("body", node)
        return node

    def render_image_from_path(self, path, name, scale=1):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.render_image(img, scale, name)

    def render_image(self, image, scale, name):
        _ = render_image_plane(self.scene, image, scale, name)

    def set_homog_group_transform(self, group_name, rotation, translation):
        # create pose matrix
        pose = np.eye(4)
        pose[:4, :4] = rotation
        pose[:3, 3] = translation

        self.set_group_pose(group_name, pose)

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

    def get_node(self, name):
        for node in self.scene.get_nodes(name=name):
            if node is not None:
                return node
        return None

    def remove_node(self, name):
        node = self.get_node(name)
        if node is None:
            return
        else:
            if self.requires_lock():
                self.viewer.render_lock.acquire()
            self.scene.remove_node(node)
            if self.requires_lock():
                self.viewer.render_lock.release()


class DefaultRenderer(Renderer):
    """Utility class for easier default renderer setup

    Args:
        Renderer ([type]): [description]
    """

    def setup(
        self,
        model,
        model_out,
        keypoints=None,
        init_keypoints=None,
        init_joints=None,
        img_path=None,
        img_scale=1,
        start=True
    ):
        if model is not None and model_out is not None:
            self.render_model(model, model_out)

        if img_path is not None:
            self.render_image_from_path(img_path, 0.5)

        self.keypoints = self.render_points(
            keypoints,
            radius=0.005,
            color=[1.0, 1.0, 1.0, 1.0])

        self.init_keypoints = self.render_keypoints(
            init_keypoints,
            radius=0.01,
            color=[1.0, 0.0, 1.0, 1.0])

        self.points = self.render_points(
            init_joints,
            radius=0.01,
            color=[0.0, 0.1, 0.0, 1.0], name="torso", group_name="body")
