import pathlib

import pytorch_volumetric as pv
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import plotly.express as px
def create_box_mesh(scale, trans):
    FILE_DIR = pathlib.Path(__file__).parent.joinpath('box_template.obj')
    mesh = o3d.io.read_triangle_mesh(str(FILE_DIR))
    v = np.array(mesh.vertices)
    v = v * np.array(scale).reshape(1, 3) + np.array(trans).reshape(1, 3)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def test_sdf():
    t_mesh = create_box_mesh((0.4, 0.4, 0.1), (0.8, 0, 0.1))
    t_mesh = trimesh.Trimesh(vertices=t_mesh.vertices, faces=t_mesh.triangles)
    f_mesh = create_box_mesh((0.75, 0.4, 0.1), (0.45, 0, -0.15))
    f_mesh = trimesh.Trimesh(vertices=f_mesh.vertices, faces=f_mesh.triangles)
    scene_mesh = trimesh.boolean.union([t_mesh, f_mesh], engine='blender')
    scene_path = pathlib.Path(__file__).parent.joinpath('scene_mesh_wrong.obj')
    trimesh.exchange.obj.export_obj(scene_mesh, scene_path)

    scene_path = pathlib.Path(__file__).parent.joinpath('scene_mesh_wrong.obj')
    # scene_path = pathlib.Path(__file__).parent.joinpath('scene_mesh_gt.obj')
    # scene_path = pathlib.Path(__file__).parent.joinpath('scene_mesh_separated.obj')
    # scene_path = pathlib.Path(__file__).parent.joinpath('scene_mesh_overlap.obj')
    scene_sdf = pv.MeshSDF(pv.MeshObjectFactory(scene_path))
    query_range = np.array([
        [-0.5, 2],
        [0, 0],
        [-0.2, 0.3],
    ])
    ret = pv.draw_sdf_slice(scene_sdf, query_range)
    v = ret[-1]
    px.imshow(v).show()
    plt.show()

if __name__ == "__main__":
    test_sdf()