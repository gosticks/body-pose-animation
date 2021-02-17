import smplx
from model import VPoserModel
import torch
import torch.nn as nn
import numpy as np


from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss


class IntersectLoss(nn.Module):
    def __init__(
        self,
        model: smplx.SMPL,
        device=torch.device('cpu'),
        dtype=torch.float32,
        batch_size=1,
        weight=1,
        sigma=0.5,
        max_collisions=8,
        point2plane=True
    ):
        """Intersections loss layer.

        Args:
            device ([type], optional): [description]. Defaults to torch.device('cpu').
            dtype ([type], optional): [description]. Defaults to torch.float32.
            weight (int, optional): Weight factor of the loss. Defaults to 1.
            sigma (float, optional): The height of the cone used to calculate the distance field loss. Defaults to 0.5.
            max_collisions (int, optional): The maximum number of bounding box collisions. Defaults to 8.
        """

        super(IntersectLoss, self).__init__()

        self.has_parameters = False

        with torch.no_grad():
            output = model(get_skin=True)
            verts = output.vertices

        face_tensor = torch.tensor(
            model.faces.astype(np.int64),
            dtype=torch.long,
            device=device) \
            .unsqueeze_(0) \
            .repeat(
                [batch_size,
                 1, 1])

        bs, nv = verts.shape[:2]
        bs, nf = face_tensor.shape[:2]

        faces_idx = face_tensor + \
            (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]

        self.register_buffer("faces_idx", faces_idx)

        # Create the search tree
        self.search_tree = BVH(max_collisions=max_collisions)

        self.pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                         point2plane=point2plane,
                                                         vectorized=True)

        # create buffer for weights
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=dtype).to(device=device)
        )

    def forward(self, pose, joints, points, keypoints, raw_output):
        verts = raw_output.vertices
        polygons = verts.view([-1, 3])[self.faces_idx]

        # find collision idx
        with torch.no_grad():
            collision_idxs = self.search_tree(polygons)

        # compute penetration loss
        return self.pen_distance(polygons, collision_idxs) * self.weight
