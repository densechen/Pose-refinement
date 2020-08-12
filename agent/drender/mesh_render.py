'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:43:00
'''
import torch
import torch.nn as nn
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.utils import (_clip_barycentric_coordinates,
                                           _interpolate_zbuf)


class MeshRendererDepth(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should be intialized with a rasterizer and shader class which each have a forward function.
    """
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then shading.

        NOTE: If the blur radius for rasterizaiton is > 0.0, some pixels can have one or more barycentric coordinates lying outside the range [0, 1]. 
        For a pixel with out of bounds barycentric coordinates with respect to a face f, clipping is required before interpolating the texture uv coordinates and z buffer so that the colors and depths are limited to the range for the corresponding face.
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        raster_setting = kwargs.get("raster_settings",
                                    self.rasterizer.raster_settings)
        if raster_setting.blur_radius > 0.0:
            # TODO: potentially move barycenteric clipping to the rasterizer.
            # If no downstream functions requiers upclipped values.
            # This will avoid uncessary re-interpolation of the z buffer.
            clipped_bary_coords = _clip_barycentric_coordinates(
                fragments.bary_coords)
            clipped_zbuf = _interpolate_zbuf(fragments.pix_to_face,
                                             clipped_bary_coords, meshes_world)
            fragments = Fragments(
                bary_coords=clipped_bary_coords,
                zbuf=clipped_zbuf,
                dists=fragments.dists,
                pix_to_face=fragments.pix_to_face,
            )
        images = self.shader(fragments, meshes_world, **kwargs)
        depth = fragments.zbuf
        return images, depth
