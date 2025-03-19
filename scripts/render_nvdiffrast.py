#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import collections
import json
from pathlib import Path
from PIL import Image
import struct

import numpy as np
import nvdiffrast.torch as dr
import torch
import torchvision

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def load_image(path, res=None, get_shape=False):
    _img = Image.open(path)
    if res is not None and res != 1:
        _img.thumbnail(
            [round(_img.width / res), round(_img.height / res)],
            Image.Resampling.LANCZOS,
        )
    img = np.array(_img).astype(np.float32) / 255
    img = torch.from_numpy(img)
    if get_shape:
        return img, _img.width, _img.height
    return img


def load_mesh(npz_path):
    mesh_data = np.load(npz_path)
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    vertex_colors = mesh_data["vertex_colors"]
    return vertices, faces, vertex_colors


def create_blend_proj_mats(camera_angle_x, img_shape, transf_mat, far, near):
    focal = img_shape[0] / (2 * np.tan(camera_angle_x / 2))
    transf_matrix = torch.tensor(transf_mat, device="cpu", dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)

    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal / img_shape[0]
    proj_matrix[1, 1] = -2 * focal / img_shape[1]
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0

    mvp_matrix = proj_matrix @ view_matrix

    return {
        "transf_matrix": transf_matrix,
        "view_matrix": view_matrix,
        "proj_matrix": proj_matrix,
        "mvp_matrix": mvp_matrix,
    }


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def create_colmap_proj_mats(focal_x, focal_y, img_shape, transf_mat, far, near):
    transf_matrix = torch.tensor(transf_mat, device="cpu", dtype=torch.float32)
    view_matrix = torch.inverse(transf_matrix)

    proj_matrix = torch.zeros(4, 4, device="cpu", dtype=torch.float32)
    proj_matrix[0, 0] = 2 * focal_x / img_shape[0]
    proj_matrix[1, 1] = -2 * focal_y / img_shape[1]
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2.0 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0

    mvp_matrix = proj_matrix @ view_matrix

    return {
        "transf_matrix": transf_matrix,
        "view_matrix": view_matrix,
        "proj_matrix": proj_matrix,
        "mvp_matrix": mvp_matrix,
    }


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def load_cameras_colmap(data_dir, res, near, far, test=False, tandt=False):
    try:
        cameras_extrinsic_file = Path(data_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = Path(data_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        raise NotImplementedError(".txt Colmap structure not supported")

    image_files = []
    for _idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        img_path = Path(data_dir, "images", extr.name)

        img = None
        if res != 1:
            height = round(height / res)
            width = round(width / res)

        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        transf_mat = np.eye(4, dtype=np.float32)
        transf_mat[:3, :3] = R
        transf_mat[:3, -1] = T

        transf_mat = np.linalg.inv(transf_mat)
        transf_mat[:3, 1:3] *= -1

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        res = 2 * res if tandt else res
        focal_x = focal_length_x / res
        focal_y = focal_length_y / res

        proj_mats = create_colmap_proj_mats(
            focal_x, focal_y, [width, height], transf_mat, far, near
        )

        image_files.append(
            {
                "name": extr.name,
                "width": width,
                "height": height,
                "img_path": str(img_path),
                "img": img,
                "focal_x": focal_x,
                "focal_y": focal_y,
                **proj_mats,
            }
        )
    sorted_imgs = sorted(image_files, key=lambda x: x["name"].split(".")[0])
    if test:
        cond = lambda x: x % 8 == 0
    else:
        cond = lambda x: x % 8 != 0
    return [c for idx, c in enumerate(sorted_imgs) if cond(idx)]


def load_cameras_blender(cameras_path, near, far, height=None, width=None):
    data_dir = Path(cameras_path).parent
    with open(cameras_path, "r") as f:
        camera_data = json.load(f)

    image_files = []
    _cam_angle_x = camera_data["camera_angle_x"]
    shape = [width, height]
    for _idx, _cam in enumerate(camera_data["frames"]):
        img_path = Path(data_dir, _cam["file_path"]).with_suffix(".png")
        if not _idx and (height is None or width is None):
            _, width, height = load_image(img_path, get_shape=True)
            shape = [width, height]
        trans_mat = np.array(_cam["transform_matrix"], dtype=np.float32)
        proj_mats = create_blend_proj_mats(_cam_angle_x, shape, trans_mat, far, near)

        image_files.append(
            {
                "name": img_path.name,
                "img_path": str(img_path),
                "width": shape[0],
                "height": shape[1],
                **proj_mats,
            }
        )
    return image_files


class MeshSplatRenderer:
    def __init__(self, vertices, faces, vertex_colors, num_layers=50):
        self.glctx = dr.RasterizeGLContext()
        self.vertices = torch.tensor(vertices, device="cuda", dtype=torch.float32)
        self.faces = torch.tensor(faces, device="cuda", dtype=torch.int32)
        self.vertex_colors = torch.tensor(
            vertex_colors, device="cuda", dtype=torch.float32
        )
        self.num_layers = num_layers

    def render(self, mvp_mat, width, height):
        pos_clip = torch.matmul(
            torch.cat([self.vertices, torch.ones_like(self.vertices[:, :1])], dim=1),
            mvp_mat.permute(0, 2, 1),
        )
        final_color = torch.zeros(
            (mvp_mat.shape[0], height, width, 4),
            device=self.vertices.device,
            dtype=self.vertices.dtype,
        )
        with dr.DepthPeeler(
            self.glctx, pos_clip, self.faces, (height, width)
        ) as peeler:
            for _ in range(self.num_layers):
                rast_out, rast_db = peeler.rasterize_next_layer()

                if rast_out is None:
                    break

                color_layer = dr.interpolate(
                    self.vertex_colors[None, ...],
                    rast_out,
                    self.faces,
                    rast_db=rast_db,
                    diff_attrs="all",
                )[0]

                alpha = color_layer[..., 3:4]
                rgb = color_layer[..., :3]
                final_color = final_color + (1.0 - final_color[..., 3:4]) * torch.cat(
                    [rgb * alpha, alpha], dim=-1
                )

        alpha = final_color[..., 3:4]
        rgb = final_color[..., :3]
        return rgb, alpha


def main(npz_path, cameras_path, near=0.01, far=100.0, dp_layers=50):
    vertices, faces, vertex_colors = load_mesh(npz_path)
    meshsplat_renderer = MeshSplatRenderer(
        vertices, faces, vertex_colors, num_layers=dp_layers
    )
    if cameras_path.endswith(".json"):
        camera_data = load_cameras_blender(cameras_path, near, far)
    else:
        is_tandt = cameras_path.name in ["truck", "train"]
        camera_data = load_cameras_colmap(cameras_path, near, far, tandt=is_tandt)

    # Render mesh
    for camera in camera_data:
        rgb, alpha = meshsplat_renderer.render(
            camera["mvp_matrix"][None, ...].to("cuda"),
            camera["width"],
            camera["height"],
        )

        print(f"Saving {camera['name']}")
        torchvision.utils.save_image(
            rgb[0].permute(2, 0, 1), f'output_{camera["name"]}.png'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render mesh using nvdiffrast")
    parser.add_argument(
        "npz_path", type=str, help="Path to the input NPZ file containing mesh data"
    )
    parser.add_argument(
        "cameras_path",
        type=str,
        help="Path to the cameras file (json) if you want to render from blender, path to the dataset if you want to render from colmap",
    )
    parser.add_argument(
        "--dp_layers", type=int, default=50, help="Number of depth peeling layers"
    )

    args = parser.parse_args()

    main(args.npz_path, args.cameras_path, args.dp_layers)
