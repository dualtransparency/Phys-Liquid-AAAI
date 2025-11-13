import gradio as gr
from typing import List, Tuple


def create_navbar() -> gr.Row:
    with gr.Row(variant="panel", elem_classes="navbar") as navbar:
        gr.Markdown("Elongevity AI Lab", elem_classes="lab-title")
    return navbar


def create_header() -> gr.Row:
    with gr.Row(variant="panel", elem_classes="header") as header:
        with gr.Column(scale=1, elem_classes="nav-button-left"):
            gr.HTML(f'<a href="http://localhost:8003" class="nav-button">上一页</a>')
        with gr.Column(scale=2, elem_classes="title-container"):
            gr.Markdown("## 空间智能 / 生成", elem_classes="section-title")
        with gr.Column(scale=1, elem_classes="nav-button-right"):
            gr.HTML(f'<a href="http://localhost:8005" class="nav-button">下一页</a>')
    return header


def create_input_panel(examples: List[str]) -> Tuple[
    gr.Image, gr.Radio, gr.ColorPicker, gr.Slider, gr.Number, gr.Slider, gr.Slider]:
    with gr.Column(variant="panel") as input_panel:
        with gr.Row():
            with gr.Column(scale=6):
                image_input = gr.Image(
                    label="上传图像",
                    image_mode="RGBA",
                    type="pil",
                    height=400,
                    elem_classes="upload-container"
                )
                gr.Markdown("提示: 请上传透明液体的分割图像")

                with gr.Row():
                    with gr.Column(scale=1):
                        background_choice = gr.Radio(
                            ["Alpha as mask", "自动去背景"],
                            value="自动去背景",
                            label="背景处理方式",
                            elem_classes="radio-group"
                        )
                        backgroud_color = gr.ColorPicker(
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
            with gr.Column(scale=6):
                examples_gallery = gr.Gallery(
                    value=examples,
                    label="示例图像",
                    columns=6,
                    height=None,
                    object_fit="contain",
                    preview=False
                )
                batch_button = gr.Button("批量生成3D模型", variant="primary")
                progress_text = gr.Textbox(label="处理进度", elem_classes="progress-text")
                stop_button = gr.Button("中止处理", variant="stop", interactive=False)
    return (
        image_input,
        background_choice,
        backgroud_color,
        foreground_ratio,
        seed,
        guidance_scale,
        step,
        examples_gallery,
        batch_button,
        progress_text,
        stop_button
    )


def create_output_panel() -> Tuple[gr.Image, gr.Image, gr.Image, gr.Model3D, gr.File, gr.File]:
    with gr.Column(variant="panel") as output_panel:
        processed_image = gr.Image(
            label="预处理图像",
            interactive=False,
            type="pil",
            image_mode="RGB",
            height=400
        )
        with gr.Row():
            image_output = gr.Image(
                label="多视角RGB图像输出",
                interactive=False,
                height=200
            )
            xyz_output = gr.Image(
                label="多视角归一化坐标输出",
                interactive=False,
                height=200
            )
        output_model = gr.Model3D(label="GLB模型", interactive=False)
        output_obj = gr.File(label="导出文件", interactive=False)
        batch_output = gr.File(label="批量导出结果", file_types=[".zip"])
    return processed_image, image_output, xyz_output, output_model, output_obj, batch_output