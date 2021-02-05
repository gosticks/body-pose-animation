
from utils.mapping import get_mapping_arr
import torch
import torch.nn.functional as F
import torch.nn as nn


class JointFilter(nn.Module):
    def __init__(
        self,
        filter_dims=3,
        mapping=None,
        model_type="smplx",
        dtype=torch.torch.float32,
        device=torch.device('cpu')
    ):
        super(JointFilter, self).__init__()

        self.dtype = dtype
        self.device = device

        if mapping is None:
            mapping = get_mapping_arr(output_format=model_type)

        # create valid joint filter
        filter = self.get_joint_filter(filter_dims, mapping)
        self.register_buffer("filter", filter)

    def get_joint_filter(self, filter_dims, mapping, threadhold=-1):
        """create a filter array for given mapping length and dims. Filter will be of shape (mapping.shape[0], filter_dims).

        Args:
            filter_dims ([type]): [description]
            mapping ([type]): list of filter mappings everything <= threadhold will be filtered  

        Returns:
            [type]: [description]
        """

        # create a list with 1s for used joints and 0 for ignored joints
        filter_shape = (len(mapping), filter_dims)

        filter = torch.zeros(
            filter_shape, dtype=self.dtype, device=self.device)
        for index, valid in enumerate(mapping > threadhold):
            if valid:
                filter[index] += 1

        return filter

    def forward(self, input):
        return input * self.filter
