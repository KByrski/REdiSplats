import argparse
import numpy as np
import os
import trimesh
import torch
from plyfile import PlyData, PlyElement
from pathlib import Path

def load_file(file_path):
    f = open(file_path, "rb")
    f.seek(0, 2)
    f.seek(0, 0)
    for line in f:
        line_str = line.decode('utf-8').strip()
        if (line_str.startswith('element vertex')):
            gaussians_num = int(line_str.split()[-1])
        if (line_str == 'end_header'):
            break
    data = np.frombuffer(f.read(), dtype=np.float32)
    data = data.reshape(data.shape[0] // 17, 17)
    return data, gaussians_num

def remove_field(data, field):
    new_names = [name for name in data.dtype.names if name != field]
    new_dtype = np.dtype([(name, data.dtype.fields[name][0]) for name in new_names])

    new_data = np.empty(data.shape, dtype=new_dtype)
    for name in new_names:
        new_data[name] = data[name]
    return new_data

def modify_ply(input_ply, output_ply, new_scales, new_rotations, new_xyz):
    with open(input_ply, 'rb') as f:
        ply_data = PlyData.read(f)

    vertex_data = np.array(ply_data['vertex'].data)
    modified_vertices = np.copy(vertex_data)

    modified_vertices['x'] = new_xyz[:, 0]
    modified_vertices['y'] = new_xyz[:, 1]
    modified_vertices['z'] = new_xyz[:, 2]

    modified_vertices['scale_0'] = new_scales[:, 0]
    modified_vertices['scale_1'] = new_scales[:, 1]

    modified_vertices['rot_0'] = new_rotations[:, 0]
    modified_vertices['rot_1'] = new_rotations[:, 1]
    modified_vertices['rot_2'] = new_rotations[:, 2]
    modified_vertices['rot_3'] = new_rotations[:, 3]

    modified_vertices = remove_field(modified_vertices, 'scale_2')

    vertex_element = PlyElement.describe(modified_vertices, 'vertex')

    with open(output_ply, 'wb') as f:
        PlyData([vertex_element], text=False).write(f)

def to_triangle(points):
  points = points.reshape(-1, 6, 3, 3)
  v0 = points[:, 0, 0]
  v1 = points[:, 1, 1]
  v1_symm = points[:, -1, 1]
  v2 = (v1 + v1_symm) / 2
  return torch.stack((v0, v1, v2), dim=1)

import torch.nn.functional as F


def rot_to_quat_batch(rot: torch.Tensor):
    """
    Implementation based on pytorch3d implementation
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot.reshape(-1, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
          F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
          ].reshape(-1, 4)

    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def get_params(triangles, Q):
  def dot(v, u):
    return (v * u).sum(dim=-1, keepdim=True)

  def proj(v, u):
    coef = dot(v, u)
    return coef * u

  v0 = triangles[:, 0]
  v1 = triangles[:, 1]
  m = triangles[:, 2]

  v_cos = v0 - m
  v_sin = v1 - m
  s1 = torch.linalg.vector_norm(v_cos, dim=-1, keepdim=True)
  s2 = torch.linalg.vector_norm(v_sin, dim=-1, keepdim=True)
  r0 = torch.cross(v_cos / s1, v_sin / s2, dim=-1)
  r0 = r0 / torch.linalg.vector_norm(r0, dim=-1, keepdim=True)
  r1 = v_cos / s1
  r2 = v_sin - proj(v_sin, r0) - proj(v_sin, r1)
  r2 = r2 / torch.linalg.vector_norm(r2, dim=-1, keepdim=True)
  s2 = dot(v_sin, r2)

  s1 = torch.log(s1 / np.sqrt(Q)) # cos(2pi / 8 * i) = 1 for i=0
  s2 = torch.log(s2 / np.sqrt(Q)) # sin(2pi / 8 * i) = 1 for i=1

  R = torch.stack([r0, r1, r2], dim=-1)
  R = rot_to_quat_batch(R)
  S = torch.cat([s1, s2], dim=-1)

  return R, S, m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert from mesh to PLY file"
    )
    parser.add_argument("--ply_path", type=str, help="Path to the original PLY file", required=True)
    parser.add_argument("--output_path", type=str, help="Path to the output PLY file", required=True)
    parser.add_argument("--frames_path", type=str, help="Path to the frames to convert", default="frames")
    parser.add_argument(
        "--quant",
        type=float,
        default=4.0,
        help="Quantile value",
    )
    
    args = parser.parse_args()
    Q = args.quant
    orig_ply_path = args.ply_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    frames = os.listdir(args.frames_path)
    for frame_name in frames:
        print("Converting", frame_name)
        frame_path = os.path.join(args.frames_path, frame_name)
        frame = trimesh.load(frame_path)
        vertices = torch.from_numpy(np.array(frame.vertices)).double()
        faces = np.array(frame.faces)
        parametrized = to_triangle(vertices[faces])
        R, S, m = get_params(parametrized, Q)    
        name = Path(frame_name).stem
        save_path = os.path.join(output_path, f"{name}.ply")
        modify_ply(orig_ply_path, save_path, S, R, m)
        print("New PLY saved at", save_path)