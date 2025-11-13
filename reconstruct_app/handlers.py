from typing import Dict, Any, Tuple
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from gradio import components
from preprocessing import preprocess_image
import gradio as gr


def register_handlers(
        model_manager: 'ModelManager',
        components_dict: Dict[str, components.Component]
):
    def handle_image_upload(img: Image.Image) -> Dict[components.Image, Dict]:
        return {components_dict['text_button']: gr.update(interactive=img is not None)}

    def handle_example_select(evt: components.SelectData) -> Image.Image:
        return Image.open(components_dict['examples'][evt.index])

    def handle_generate_click(
            image: Image.Image,
            bg_choice: str,
            fg_ratio: float,
            bg_color: str,
            seed: int,
            scale: float,
            step: int
    ) -> Tuple[Image.Image, Image.Image, Image.Image, str, str]:
        processed = preprocess_image(
            image,
            bg_choice,
            fg_ratio,
            tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        )
        stage1, stage2, glb, obj = model_manager.generate_3d(
            processed,
            seed,
            scale,
            step
        )
        return processed, stage1, stage2, glb, obj

    def batch_process(
            bg_choice: str,
            fg_ratio: float,
            bg_color: str,
            seed: int,
            scale: float,
            step: int,
            processing_flag: bool,
            processed_results: list
    ):
        if not processing_flag:
            return
        total = len(components_dict['examples'])
        results = []
        try:
            for i, img_path in enumerate(components_dict['examples']):
                if not processing_flag:
                    break
                img = Image.open(img_path).convert("RGBA")
                processed = preprocess_image(
                    img,
                    bg_choice,
                    fg_ratio,
                    tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                )
                stage1, stage2, glb, obj = model_manager.generate_3d(
                    processed,
                    seed,
                    scale,
                    step
                )
                results.append((glb, obj))
                yield {
                    components_dict['processed_image']: processed,
                    components_dict['image_output']: stage1,
                    components_dict['xyz_output']: stage2,
                    components_dict['output_model']: glb,
                    components_dict['output_obj']: obj,
                    components_dict['progress_text']: f"处理中：{i + 1}/{total}",
                    components_dict['text_button']: gr.update(interactive=False),
                    components_dict['stop_button']: gr.update(interactive=True)
                }
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "results.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for glb, obj in results:
                        zipf.write(glb, Path(glb).name)
                        zipf.write(obj, Path(obj).name)
                yield {
                    components_dict['batch_output']: str(zip_path),
                    components_dict['progress_text']: f"处理完成，共生成{len(results)}个文件"
                }
        finally:
            yield {
                components_dict['processing_flag']: False,
                components_dict['text_button']: gr.update(interactive=True)
            }

    def handle_stop(processing_flag: bool) -> Dict:
        return {
            components_dict['processing_flag']: False,
            components_dict['progress_text']: "处理已中止"
        }

    # 绑定事件处理
    components_dict['image_input'].change(
        handle_image_upload,
        inputs=components_dict['image_input'],
        outputs=components_dict['text_button']
    )

    components_dict['examples_gallery'].select(
        handle_example_select,
        outputs=components_dict['image_input']
    )

    components_dict['text_button'].click(
        handle_generate_click,
        inputs=[
            components_dict['image_input'],
            components_dict['background_choice'],
            components_dict['foreground_ratio'],
            components_dict['backgroud_color'],
            components_dict['seed'],
            components_dict['guidance_scale'],
            components_dict['step']
        ],
        outputs=[
            components_dict['processed_image'],
            components_dict['image_output'],
            components_dict['xyz_output'],
            components_dict['output_model'],
            components_dict['output_obj']
        ]
    )

    components_dict['batch_button'].click(
        lambda: (True, []),
        outputs=[components_dict['processing_flag'], components_dict['processed_results']]
    ).then(
        batch_process,
        inputs=[
            components_dict['background_choice'],
            components_dict['foreground_ratio'],
            components_dict['backgroud_color'],
            components_dict['seed'],
            components_dict['guidance_scale'],
            components_dict['step'],
            components_dict['processing_flag'],
            components_dict['processed_results']
        ],
        outputs=[
            components_dict['processed_image'],
            components_dict['image_output'],
            components_dict['xyz_output'],
            components_dict['output_model'],
            components_dict['output_obj'],
            components_dict['batch_output'],
            components_dict['progress_text'],
            components_dict['text_button'],
            components_dict['stop_button']
        ]
    )

    components_dict['stop_button'].click(
        handle_stop,
        inputs=components_dict['processing_flag'],
        outputs=[
            components_dict['processing_flag'],
            components_dict['progress_text']
        ]
    )