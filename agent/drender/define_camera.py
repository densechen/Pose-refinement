'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:42:52
'''
import random

import numpy as np
import torch
from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                PointLights, RasterizationSettings,
                                SoftSilhouetteShader, TexturedSoftPhongShader)

from .mesh_render import MeshRendererDepth
from .real_camera import OpenGLRealPerspectiveCameras


def define_camera(image_size=640,
                  image_height=480,
                  image_width=640,
                  fx=500,
                  fy=500,
                  cx=320,
                  cy=240,
                  device="cuda:0"):
    # define camera
    cameras = OpenGLRealPerspectiveCameras(
        focal_length=((fx, fy), ),  # Nx2
        principal_point=((cx, cy), ),  # Nx2
        x0=0,
        y0=0,
        w=image_size,
        h=image_size,
        znear=0.0001,
        zfar=100.0,
        device=device)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    phong_raster_settings = RasterizationSettings(image_size=image_size,
                                                  blur_radius=0.0,
                                                  faces_per_pixel=1)

    # We can add a point light in front of the object.
    # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    phong_renderer = MeshRendererDepth(
        rasterizer=MeshRasterizer(cameras=cameras,
                                  raster_settings=phong_raster_settings),
        shader=TexturedSoftPhongShader(device=device)).to(device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size 640x640. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py for an explanation of this parameter.
    silhouette_raster_settings = RasterizationSettings(
        image_size=image_size,  # longer side or scaled longer side
        # blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        blur_radius=0.0,
        # The nearest faces_per_pixel points along the z-axis.
        faces_per_pixel=1)

    # Create a silhouette mesh renderer by composing a rasterizer and a shader
    silhouete_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
                                  raster_settings=silhouette_raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)).to(device)
    return phong_renderer, silhouete_renderer
