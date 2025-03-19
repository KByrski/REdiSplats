import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_file(file_path):
    f = open(file_path, "rb")
    f.seek(0, 2)
    f.seek(0, 0)
    for line in f:
        line_str = line.decode("utf-8").strip()
        if line_str.startswith("element vertex"):
            gaussians_num = int(line_str.split()[-1])
        if line_str == "end_header":
            break
    data = np.frombuffer(f.read(), dtype=np.float32)
    data = data.reshape(data.shape[0] // 17, 17)
    return data, gaussians_num


def construct_list_of_attributes():
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(3):
        l.append("f_dc_{}".format(i))
    l.append("opacity")
    for i in range(3):
        l.append("scale_{}".format(i))
    for i in range(4):
        l.append("rot_{}".format(i))
    return l


def save_ply(out_path, xyz, scale, rotation, rgb, opacity):
    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, rgb, opacity, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)


def parse_gaussians(gaussian_data, gaussians_num):
    raw_xyz = gaussian_data[:, 0:3]
    xyz = np.expand_dims(raw_xyz, 1)
    xyz = np.expand_dims(xyz, 1)
    xyz = np.repeat(xyz, 8, 1)
    xyz = xyz.reshape(gaussians_num, 8, 3, 1)

    raw_scales = gaussian_data[:, 10:13]
    scale = np.expand_dims(np.exp(raw_scales), 1)
    scale = np.expand_dims(scale, 1)
    scale = np.repeat(scale, 8, 1)
    scale = scale.reshape(gaussians_num, 8, 3, 1)

    raw_quaternions = gaussian_data[:, 13:17]
    a = np.expand_dims(gaussian_data[:, 13], 1)
    b = np.expand_dims(gaussian_data[:, 14], 1)
    c = np.expand_dims(gaussian_data[:, 15], 1)
    d = np.expand_dims(gaussian_data[:, 16], 1)
    norm = np.sqrt(a**2 + b**2 + c**2 + d**2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    bb, cc, dd = b**2, c**2, d**2
    ab, ac, ad = a * b, a * c, a * d
    bb, bc, bd = bb, b * c, b * d
    cc, cd, dd = cc, c * d, dd
    R11, R12, R13 = 1.0 - 2 * (cc + dd), 2 * (bc - ad), 2 * (bd + ac)
    R21, R22, R23 = 2 * (bc + ad), 1.0 - 2 * (bb + dd), 2 * (cd - ab)
    R31, R32, R33 = 2 * (bd - ac), 2 * (cd + ab), 1.0 - 2 * (bb + cc)
    R1 = np.expand_dims(np.concatenate((R11, R12, R13), 1), 1)
    R2 = np.expand_dims(np.concatenate((R21, R22, R23), 1), 1)
    R3 = np.expand_dims(np.concatenate((R31, R32, R33), 1), 1)
    R = np.expand_dims(np.concatenate((R1, R2, R3), 1), 1)
    R = np.repeat(R, 8, 1)

    raw_colors = gaussian_data[:, 6:9]
    colors = np.clip((0.28209479177387814 * raw_colors) + 0.5, 0, 1)
    raw_alpha = gaussian_data[:, 9:10]
    alpha = 1.0 / (1.0 + np.exp(-raw_alpha))
    colors_opac = np.concatenate((colors, alpha), 1)
    return (
        xyz,
        scale,
        R,
        colors_opac,
        raw_xyz,
        raw_scales,
        raw_quaternions,
        raw_colors,
        raw_alpha,
    )


def gaussians_to_meshsplat(gauss_xyz, scales, R, colors_opac, gaussians_num, quantile):
    gauss_as_polygon_vertices_x = np.expand_dims(np.zeros(8), 1)
    gauss_as_polygon_vertices_y = np.expand_dims(
        np.cos(((2.0 * np.pi) / 8) * np.arange(8)) * np.sqrt(quantile), 1
    )
    gauss_as_polygon_vertices_z = np.expand_dims(
        np.sin(((2.0 * np.pi) / 8) * np.arange(8)) * np.sqrt(quantile), 1
    )
    gauss_as_polygon_vertices = np.concatenate(
        (
            gauss_as_polygon_vertices_x,
            gauss_as_polygon_vertices_y,
            gauss_as_polygon_vertices_z,
        ),
        1,
    )
    gauss_as_polygon_vertices = np.expand_dims(gauss_as_polygon_vertices, 0)

    gauss_as_polygon_indices_p1 = np.expand_dims(np.zeros(6), 1)
    gauss_as_polygon_indices_p2 = np.expand_dims(np.arange(6) + 1, 1)
    gauss_as_polygon_indices_p3 = np.expand_dims(np.arange(6) + 2, 1)
    gauss_as_polygon_indices = np.concatenate(
        (
            gauss_as_polygon_indices_p1,
            gauss_as_polygon_indices_p2,
            gauss_as_polygon_indices_p3,
        ),
        1,
    )
    gauss_as_polygon_indices = np.expand_dims(gauss_as_polygon_indices, 0)

    gaussians_as_polygons_vertices = np.repeat(
        gauss_as_polygon_vertices, gaussians_num, 0
    )
    gaussians_as_polygons_vertices = np.expand_dims(
        gaussians_as_polygons_vertices.reshape(gaussians_num, 8, 3), 3
    )
    gaussians_as_polygons_vertices = (
        np.matmul(R, gaussians_as_polygons_vertices * scales) + gauss_xyz
    )
    gaussians_as_polygons_vertices = gaussians_as_polygons_vertices.reshape(
        gaussians_num * 8, 3
    )

    gaussians_as_polygons_indices = np.repeat(
        gauss_as_polygon_indices, gaussians_num, 0
    )
    gaussians_as_polygons_indices = gaussians_as_polygons_indices + (
        np.expand_dims(np.expand_dims(np.arange(gaussians_num), 1), 2) * 8
    )
    gaussians_as_polygons_indices = gaussians_as_polygons_indices.astype(int)
    gaussians_as_polygons_indices = gaussians_as_polygons_indices.reshape(
        gaussians_num * 6 * 3
    )

    colors = colors_opac
    colors = np.repeat(colors, 6, 0)

    triangles = gaussians_as_polygons_vertices[gaussians_as_polygons_indices]
    triangles = triangles.reshape(gaussians_num * 6, 3, 3)

    return gaussians_as_polygons_vertices, gaussians_as_polygons_indices, colors


def main(file_path, opac_threshold=0, quant=4.0):
    assert (
        opac_threshold >= 0 and opac_threshold <= 1
    ), "opac_threshold must be between 0 and 1"

    file_path = Path(file_path)
    masked_ply_out_path = file_path.parent / "masked.ply"
    new_output_path = file_path.parent / "meshsplat.npz"
    gaussian_data, gaussians_num = load_file(file_path)

    (
        gauss_xyz,
        scales,
        R,
        colors_opac,
        raw_xyz,
        raw_scales,
        raw_quaternions,
        raw_colors,
        raw_alpha,
    ) = parse_gaussians(gaussian_data, gaussians_num)

    # do mask on alpha
    if opac_threshold > 0:
        mask = colors_opac[:, 3] > opac_threshold

        gauss_xyz = gauss_xyz[mask]
        scales = scales[mask]
        R = R[mask]
        colors_opac = colors_opac[mask]
        gaussians_num = gauss_xyz.shape[0]

        raw_xyz = raw_xyz[mask]
        raw_scales = raw_scales[mask]
        raw_quaternions = raw_quaternions[mask]
        raw_colors = raw_colors[mask]
        raw_alpha = raw_alpha[mask]

    save_ply(
        masked_ply_out_path, raw_xyz, raw_scales, raw_quaternions, raw_colors, raw_alpha
    )

    gaussians_as_polygons_vertices, gaussians_as_polygons_indices, colors = (
        gaussians_to_meshsplat(gauss_xyz, scales, R, colors_opac, gaussians_num, quant)
    )

    meshsplat_verts = gaussians_as_polygons_vertices
    meshsplat_faces = gaussians_as_polygons_indices.reshape(-1, 3)
    meshsplat_colors = np.repeat(colors_opac, 8, 0)

    np.savez(
        new_output_path,
        vertices=meshsplat_verts,
        faces=meshsplat_faces,
        vertex_colors=meshsplat_colors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mesh visualization from PLY file"
    )
    parser.add_argument("ply_path", type=str, help="Path to the input PLY file")
    parser.add_argument(
        "--opac_threshold",
        type=float,
        default=0.5,
        help="Opacity threshold for the mesh",
    )
    parser.add_argument(
        "--quant",
        type=float,
        default=4.0,
        choices=[4.0, 6.2514, 7.8147, 11.3449, 21.1075],
        help="Quantile value",
    )

    args = parser.parse_args()
    main(args.ply_path, args.opac_threshold, args.quant)
