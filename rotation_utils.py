import torch
import numpy as np

def quat_to_rotmat(q):
    """
    Convert quaternions (x, y, z, w) to 3x3 rotation matrices.
    q: tensor of shape (..., 4)
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x + x, y + y, z + z
    xx, yx, yy = x * x2, y * x2, y * y2
    zx, zy, zz = z * x2, z * y2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    
    m = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    m[..., 0, 0] = 1.0 - yy - zz
    m[..., 0, 1] = yx - wz
    m[..., 0, 2] = zx + wy
    
    m[..., 1, 0] = yx + wz
    m[..., 1, 1] = 1.0 - xx - zz
    m[..., 1, 2] = zy - wx
    
    m[..., 2, 0] = zx - wy
    m[..., 2, 1] = zy + wx
    m[..., 2, 2] = 1.0 - xx - yy
    return m

def rotmat_to_quat(m):
    """
    Convert 3x3 rotation matrices to quaternions (x, y, z, w).
    m: tensor of shape (..., 3, 3)
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    
    trace = m00 + m11 + m22
    q = torch.empty(m.shape[:-2] + (4,), dtype=m.dtype, device=m.device)
    
    # Handle trace > 0
    cond1 = trace > 0
    s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2.0
    q[cond1, 3] = 0.25 * s1[cond1]
    q[cond1, 0] = (m21[cond1] - m12[cond1]) / s1[cond1]
    q[cond1, 1] = (m02[cond1] - m20[cond1]) / s1[cond1]
    q[cond1, 2] = (m10[cond1] - m01[cond1]) / s1[cond1]
    
    # Handle m00 is largest
    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 2.0
    q[cond2, 3] = (m21[cond2] - m12[cond2]) / s2[cond2]
    q[cond2, 0] = 0.25 * s2[cond2]
    q[cond2, 1] = (m01[cond2] + m10[cond2]) / s2[cond2]
    q[cond2, 2] = (m02[cond2] + m20[cond2]) / s2[cond2]
    
    # Handle m11 is largest
    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    s3 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-8)) * 2.0
    q[cond3, 3] = (m02[cond3] - m20[cond3]) / s3[cond3]
    q[cond3, 0] = (m01[cond3] + m10[cond3]) / s3[cond3]
    q[cond3, 1] = 0.25 * s3[cond3]
    q[cond3, 2] = (m12[cond3] + m21[cond3]) / s3[cond3]
    
    # Handle m22 is largest
    cond4 = (~cond1) & (~cond2) & (~cond3)
    s4 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-8)) * 2.0
    q[cond4, 3] = (m10[cond4] - m01[cond4]) / s4[cond4]
    q[cond4, 0] = (m02[cond4] + m20[cond4]) / s4[cond4]
    q[cond4, 1] = (m12[cond4] + m21[cond4]) / s4[cond4]
    q[cond4, 2] = 0.25 * s4[cond4]
    
    return q

def quat_to_6d(q):
    """
    Convert (x, y, z, w) to 6D continuous representation.
    Takes first two columns of the rotation matrix.
    """
    m = quat_to_rotmat(q)
    return m[..., :, :2].reshape(q.shape[:-1] + (6,))

def rot6d_to_rotmat(x):
    """
    Convert 6D representation to 3x3 rotation matrix using Gram-Schmidt.
    x: tensor of shape (..., 6)
    """
    x = x.view(x.shape[:-1] + (3, 2))
    a1 = x[..., :, 0]
    a2 = x[..., :, 1]
    
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack((b1, b2, b3), dim=-1)

def rot6d_to_quat(x):
    """
    Convert 6D representation back to (x, y, z, w) quaternion.
    """
    m = rot6d_to_rotmat(x)
    return rotmat_to_quat(m)
