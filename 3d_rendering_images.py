import os
import numpy as np
import pywavefront
import pyrr
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

# 初始化 GLFW，创建 OpenGL 上下文
def init_glfw(width, height, title):
    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)
    return window

# 渲染场景
def render_scene(obj_model, width, height, view_matrix):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-2, 2, -2, 2, -10, 10)  # 正交投影

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glLoadMatrixf(view_matrix)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    # 设置光照
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    ambient_light = [0.2, 0.2, 0.2, 1.0]
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
    diffuse_light = [0.8, 0.8, 0.8, 1.0]
    light_position = [1.0, 1.0, 1.0, 0.0]  # 方向光
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    # 启用颜色追踪
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # 渲染 OBJ 模型
    glColor3f(1.0, 1.0, 1.0)  # 设置模型颜色为白色
    for mesh in obj_model.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3fv(obj_model.vertices[vertex_i])
        glEnd()

# 保存帧缓冲区为图像
def save_framebuffer(width, height, filename):
    glReadBuffer(GL_FRONT)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 翻转图像
    image.save(filename)

# 主渲染函数
def render_orthogonal_views(obj_filename, output_prefix, width=512, height=512):
    # 初始化 GLFW 和 OpenGL
    window = init_glfw(width, height, "OBJ Renderer")

    # 加载 OBJ 文件
    obj_model = pywavefront.Wavefront(obj_filename, collect_faces=True)

    # 定义六个视角的视图矩阵
    views = {
        'front': pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 2]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0])),
        'back': pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, -2]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0])),
        'left': pyrr.matrix44.create_look_at(pyrr.Vector3([-2, 0, 0]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0])),
        'right': pyrr.matrix44.create_look_at(pyrr.Vector3([2, 0, 0]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0])),
        'top': pyrr.matrix44.create_look_at(pyrr.Vector3([0, 2, 0]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 0, -1])),
        'bottom': pyrr.matrix44.create_look_at(pyrr.Vector3([0, -2, 0]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 0, 1])),
    }

    # 渲染每个视角并保存图像
    for view_name, view_matrix in views.items():
        render_scene(obj_model, width, height, view_matrix)
        save_framebuffer(width, height, f"{output_prefix}_{view_name}.png")
        print(f"Saved {output_prefix}_{view_name}.png")

    # 关闭窗口
    glfw.terminate()

# 主函数入口
if __name__ == "__main__":
    obj_filename = r'E:\datasets\F0016OL4S2R2_0039_003_new\fluid_mesh_0033.obj'  # 替换为你的 .obj 文件路径
    output_prefix = 'render_output'  # 输出图片的前缀
    render_orthogonal_views(obj_filename, output_prefix)
