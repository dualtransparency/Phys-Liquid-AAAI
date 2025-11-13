import numpy as np
import gradio as gr
from omegaconf import OmegaConf
import torch
from PIL import Image
import PIL
from pipelines import TwoStagePipeline
import rembg
import os
import json
import argparse

from model import CRM
from inference import generate3d

pipeline = None
rembg_session = rembg.new_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def remove_background(
        image: PIL.Image.Image,
        rembg_session=None,
        force: bool = False,
        **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alpha channel not empty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = (
            (original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image


def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


def preprocess_image(image, background_choice, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force_remove=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")


def gen_image(input_image, seed, scale, step):
    # input_image: 256*256 rgb
    global pipeline, model, args
    pipeline.set_seed(seed)
    rt_dict = pipeline(input_image, scale=scale, step=step)
    stage1_images = rt_dict["stage1_images"]  # list size: 6, Image: 256*256
    stage2_images = rt_dict["stage2_images"]  # list size: 6, Image: 256*256
    np_imgs = np.concatenate(stage1_images, 1)  # 6 * (256, 256, 3) -> (256, 1536, 3)
    np_xyzs = np.concatenate(stage2_images, 1)  # 6 * (256, 256, 3) -> (256, 1536, 3)
    glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, args.device)
    return Image.fromarray(np_imgs), Image.fromarray(np_xyzs), glb_path, obj_path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--stage1_config",
    type=str,
    default="configs/nf7_v3_SNR_rd_size_stroke.yaml",
    help="config for stage1",
)
parser.add_argument(
    "--stage2_config",
    type=str,
    default="configs/stage2-v2-snr.yaml",
    help="config for stage2",
)

parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# crm_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="CRM.pth")
crm_path = "/mnt/ssd/fyz/CRM/CRM.pth"

specs = json.load(open("configs/specs_objaverse_total.json"))
model = CRM(specs).to(args.device)
model.load_state_dict(torch.load(crm_path, map_location=args.device), strict=False)

stage1_config = OmegaConf.load(args.stage1_config).config
stage2_config = OmegaConf.load(args.stage2_config).config
stage2_sampler_config = stage2_config.sampler
stage1_sampler_config = stage1_config.sampler

stage1_model_config = stage1_config.models
stage2_model_config = stage2_config.models

# xyz_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="ccm-diffusion.pth")
# pixel_path = hf_hub_download(repo_id="Zhengyi/CRM", filename="pixel-diffusion.pth")
xyz_path = "/mnt/ssd/fyz/CRM/ccm-diffusion.pth"
pixel_path = "/mnt/ssd/fyz/CRM/pixel-diffusion.pth"
# pixel_path = "/workspace/CRM/train_logs/nf7_v3_SNR_rd_size_stroke_train-default-2024-11-05T09-49-50/ckpts/unet-1008.pth"

stage1_model_config.resume = pixel_path
stage2_model_config.resume = xyz_path

pipeline = TwoStagePipeline(
    stage1_model_config,
    stage2_model_config,
    stage1_sampler_config,
    stage2_sampler_config,
    device=args.device,
    dtype=torch.float16
)

css = "/home/fyz/CRM/styles.css"
with open(css, 'r', encoding='utf-8') as file:
    css_content = file.read()

# 定义示例图片和技术原理图片目录
examples_dir = "/home/fyz/CRM/liquid_examples"
mechanism_dir = "/home/fyz/sam2/mechanism_examples"

prev_url = "http://localhost:8003"  # 上一页的目标地址
next_url = "http://localhost:8005"  # 下一页的目标地址

# 加载示例图片和技术原理图片
example_images = [
    os.path.join(examples_dir, img)
    for img in os.listdir(examples_dir)
    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
]
mechanism_images = [
    os.path.join(mechanism_dir, img)
    for img in os.listdir(mechanism_dir)
    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
]

with gr.Blocks(css=css_content, theme="default", elem_classes="gradio-app") as demo:
    # 页面标题
    with gr.Row(variant="panel", elem_classes="navbar"):
        gr.Markdown("Elongevity AI Lab", elem_classes="lab-title")

    # 页面头部 - 标题部分
    with gr.Row(variant="panel", elem_classes="header"):
        # 左侧按钮（上一页）
        with gr.Column(scale=1, elem_classes="nav-button-left"):
            gr.HTML(f'<a href="{prev_url}" class="nav-button" style="text-decoration: none;" target="_self"">上一页</a>')
        # 中间标题
        with gr.Column(scale=2, elem_classes="title-container"):
            gr.Markdown("## 空间智能 / 生成", elem_classes="section-title")
        # 右侧按钮（下一页）
        with gr.Column(scale=1, elem_classes="nav-button-right"):
            gr.HTML(f'<a href="{next_url}" class="nav-button" style="text-decoration: none;" target="_self"">下一页</a>')
    with gr.Column(variant="panel"):
        gr.Markdown("# 基于物理规则的双透明液体三维重建", elem_classes="center-title")

    # 技术原理展示
    with gr.Column(variant="panel"):
        gr.Gallery(
            value=mechanism_images,
            label="技术原理示意图",
            columns=3,
            height=None,
            object_fit="contain",
            preview=True,
            # elem_classes="scrollable-gallery"
        )
        gr.Markdown("# 技术应用", elem_classes="center-title")

    # 输入区域
    with gr.Column(variant="panel", elem_classes="center-title"):
        gr.Markdown("## 模型输入", elem_classes="center-title")

    with gr.Row(variant="panel", elem_classes="input-container"):
        with gr.Column(scale=6):
            # 上传区
            with gr.Column(variant="panel"):
                image_input = gr.Image(
                    label="上传图像",
                    image_mode="RGBA",
                    type="pil",
                    height=400,
                    elem_classes="upload-container"
                )
                gr.Markdown("提示: 请上传透明液体的分割图像")
            with gr.Column(variant="panel"):
                with gr.Row():
                    with gr.Column(scale=1):
                        background_choice = gr.Radio(
                            ["Alpha as mask", "自动去背景"],
                            value="自动去背景",
                            label="背景处理方式",
                            elem_classes="radio-group"
                        )
                        back_groud_color = gr.ColorPicker(
                            label="背景颜色",
                            value="#7F7F7F",
                            interactive=False,
                            elem_classes="color-picker"
                        )
                        foreground_ratio = gr.Slider(
                            label="前景比例",
                            minimum=0.5,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            elem_classes="slider"
                        )
                    with gr.Column(scale=1):
                        seed = gr.Number(
                            value=1234,
                            label="随机种子",
                            precision=0,
                            elem_classes="number-input"
                        )
                        guidance_scale = gr.Slider(
                            value=5.5,
                            minimum=3,
                            maximum=10,
                            step=0.5,
                            label="引导比例",
                            elem_classes="slider"
                        )
                        step = gr.Slider(
                            value=50,
                            minimum=30,
                            maximum=100,
                            step=1,
                            label="采样步数",
                            elem_classes="slider"
                        )
        # 参数设置区
        with gr.Column(scale=6, variant="panel"):
            # 示例区
            gr.Markdown("## 示例图像", elem_classes="center-title")
            examples_gallery = gr.Gallery(
                value=example_images,
                label="示例图像",
                columns=6,
                height=None,
                object_fit="contain",
                preview=False,
                # elem_classes="scrollable-gallery"
            )

    # 运行按钮（居中显示）
    with gr.Row(variant="panel"):
        text_button = gr.Button("生成3D模型",
                                variant="primary",
                                interactive=False,
                                elem_classes=["button"])
    with gr.Row(variant="panel", elem_classes="header"):
        gr.Markdown("## 模型输出", elem_classes="center-title")
    with gr.Column(variant="panel", elem_classes="output-container"):
        # 主布局分为左右两列
        with gr.Row():
            # 左侧：预处理图像
            with gr.Column(scale=2, min_width=400):
                processed_image = gr.Image(
                    label="预处理图像",
                    interactive=False,
                    type="pil",
                    image_mode="RGB",
                    height=400,
                    elem_classes="output-container-inner"
                )
                output_obj = gr.File(
                    interactive=False,
                    label="导出文件",
                    elem_classes="secondary small-download-box"
                )

            # 右侧：分为上下两部分
            with gr.Column(scale=3):
                # 右上部分：RGB 输出 + CCM 输出（上下排列）
                with gr.Column():
                    image_output = gr.Image(
                        interactive=False,
                        label="多视角RGB图像输出",
                        height=200,
                        elem_classes="output-container-inner"
                    )
                    xyz_output = gr.Image(
                        interactive=False,
                        label="多视角归一化坐标输出",
                        height=200,
                        elem_classes="output-container-inner"
                    )

                # 右下部分：3D 模型输出
                with gr.Row():
                    with gr.Column():
                        output_model = gr.Model3D(
                            label="GLB模型",
                            interactive=False,
                            elem_classes="model3d-output"
                        )

    # 页脚
    with gr.Column():
        gr.HTML("<hr>")
        gr.Markdown("Elongevity AI Lab ©️ 2025", elem_classes="footer-text")

    # 状态存储
    state = gr.State()


    # 交互逻辑（保持原有逻辑，仅调整选择器名称）
    def toggle_button(img):
        return gr.update(interactive=img is not None)

    image_input.change(toggle_button, image_input, text_button)

    def select_example(evt: gr.SelectData):
        return example_images[evt.index]

    examples_gallery.select(select_example, None, image_input)

    def start_processing():
        return gr.update(value="推理中...", interactive=False, elem_classes=["button", "loading"])


    def finish_processing():
        return gr.update(value="生成3D模型", interactive=True, elem_classes=["button"])

    inputs = [
        processed_image,
        seed,
        guidance_scale,
        step,
    ]
    outputs = [
        image_output,
        xyz_output,
        output_model,
        output_obj,
    ]

    text_button.click(fn=start_processing, inputs=None, outputs=[text_button]).then(
        fn=check_input_image, inputs=[image_input]
    ).success(
        fn=preprocess_image,
        inputs=[image_input, background_choice, foreground_ratio, back_groud_color],
        outputs=[processed_image]
    ).success(
        fn=gen_image,
        inputs=inputs,
        outputs=outputs
    ).then(
        fn=finish_processing, inputs=None, outputs=[text_button]
    )

demo.queue().launch(server_name="0.0.0.0",server_port=8004)
