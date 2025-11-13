import os
from config import Config
from model_utils import ModelManager
from ui_components import (
    create_navbar,
    create_header,
    create_input_panel,
    create_output_panel
)
from handlers import register_handlers
import gradio as gr


def main():
    # 初始化配置
    cfg = Config()

    # 初始化模型管理器
    model_manager = ModelManager(cfg)
    model_manager.load_models()

    # 创建Gradio界面
    with gr.Blocks(css=open("assets/styles.css").read(), theme="default") as demo:
        # 顶部导航栏
        create_navbar()

        # 页面标题栏
        create_header()

        # 技术原理展示区
        with gr.Column(variant="panel"):
            gr.Markdown("# 基于物理规则的双透明液体三维重建", elem_classes="center-title")

            gr.Gallery(
                value=[os.path.join(cfg.mechanism_dir, img) for img in os.listdir(cfg.mechanism_dir)
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))],
                label="技术原理示意图",
                columns=3,
                object_fit="contain"
            )
            gr.Markdown("# 技术应用", elem_classes="center-title")

        # 输入输出面板
        with gr.Row(variant="panel", elem_classes="input-container"):
            # 输入组件
            (
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
            ) = create_input_panel([os.path.join(cfg.examples_dir, img)
                                    for img in os.listdir(cfg.examples_dir)
                                    if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

            # 输出组件
            processed_image, image_output, xyz_output, output_model, output_obj, batch_output = create_output_panel()

        # 生成按钮
        text_button = gr.Button("生成3D模型", variant="primary", elem_classes=["button"])

        # 注册事件处理
        register_handlers({
            "image_input": image_input,
            "background_choice": background_choice,
            "backgroud_color": backgroud_color,
            "foreground_ratio": foreground_ratio,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "step": step,
            "examples": [os.path.join(cfg.examples_dir, img)
                         for img in os.listdir(cfg.examples_dir)
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))],
            "text_button": text_button,
            "batch_button": batch_button,
            "processed_image": processed_image,
            "image_output": image_output,
            "xyz_output": xyz_output,
            "output_model": output_model,
            "output_obj": output_obj,
            "batch_output": batch_output,
            "progress_text": progress_text,
            "stop_button": stop_button,
            "processing_flag": gr.State(False),
            "processed_results": gr.State([])
        }, model_manager)

    # 启动服务
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=8004,
        share=False,
        show_api=False
    )


if __name__ == "__main__":
    main()