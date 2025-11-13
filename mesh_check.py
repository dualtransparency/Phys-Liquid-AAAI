import trimesh
import matplotlib.pyplot as plt


def display_mesh(mesh, title="3D Mesh View"):
    """显示 Trimesh 对象的 3D 网格"""
    # 获取网格的顶点和面
    vertices = mesh.vertices
    faces = mesh.faces

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制网格
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], cmap='viridis', edgecolor='none')

    # 设置图形标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置标题
    ax.set_title(title)

    # 显示图形
    plt.show()


def display_obj(obj_file_path):
    """加载并显示 .obj 文件的 3D 网格"""
    # 加载 .obj 文件
    mesh = trimesh.load(obj_file_path)
    display_mesh(mesh, title=f"3D View of {obj_file_path}")


def clean_and_display_obj_by_volume(obj_file_path):
    """根据体积保留最大的闭合部分，并显示处理后的网格"""
    # 加载 .obj 文件
    mesh = trimesh.load(obj_file_path)

    # 分割 mesh 为多个部分
    meshes = mesh.split()

    # 找到最大的部分（根据体积）
    largest_mesh = max(meshes, key=lambda m: m.volume)

    # 显示处理后的 mesh
    display_mesh(largest_mesh, title="Cleaned 3D Mesh (Largest by Volume)")


if __name__ == "__main__":
    obj_file_path = "/workspace/data2/multiview_data/F0009OL8S3R5/F0009OL8S3R5_0041/output3d.obj"

    # 显示处理前的 .obj 文件
    print("Displaying original mesh...")
    display_obj(obj_file_path)

    # 显示处理后的 .obj 文件（根据体积保留最大部分）
    print("Displaying cleaned mesh by volume...")
    clean_and_display_obj_by_volume(obj_file_path)
