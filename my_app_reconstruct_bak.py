# -*- coding: utf-8 -*-
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
import time
import math # For ceiling function

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

# --- Constants ---
BATCH_SIZE = 20
RESULTS_SAVE_DIR = Path("/home/fyz/CRM/restruct_examples/restruct_results")
LOG_TEXTBOX_LINES = 10

# --- Default Processing Parameters ---
DEFAULT_BACKGROUND_CHOICE = "自动去背景" # Options: "Alpha as mask", "自动去背景"
DEFAULT_BACKGROUND_COLOR = (127, 127, 127) # Grey background (RGB tuple)
DEFAULT_FOREGROUND_RATIO = 1.0
DEFAULT_SEED = 1234 # Base seed, will increment per image
DEFAULT_GUIDANCE_SCALE = 5.5
DEFAULT_STEP = 50

# --- Initialization ---
pipeline = None
rembg_session = rembg.new_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {RESULTS_SAVE_DIR}")
print(f"Using Default Parameters: BG Remove='{DEFAULT_BACKGROUND_CHOICE}', FG Ratio={DEFAULT_FOREGROUND_RATIO}, Scale={DEFAULT_GUIDANCE_SCALE}, Steps={DEFAULT_STEP}")

# --- Helper Functions ---
# ... (expand_to_square, remove_background, do_resize_content, add_background - no changes needed in function code) ...
def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    width, height = image.size
    if width == height: return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def remove_background(image: PIL.Image.Image, rembg_session=None, force: bool = False, **rembg_kwargs) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    if force: do_remove = True
    if do_remove:
        try: image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
        except Exception as e: print(f"Error during rembg.remove: {e}")
    return image

def do_resize_content(original_image: Image, scale_rate):
    if scale_rate != 1:
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        if new_size[0] == 0 or new_size[1] == 0: return original_image
        resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else: return original_image

def add_background(image, bg_color=(255, 255, 255)):
    background = Image.new("RGBA", image.size, bg_color + (255,)) # Add alpha for RGBA background
    if image.mode != 'RGBA': image = image.convert('RGBA')
    return Image.alpha_composite(background, image)

# --- Core Processing Functions (Using Defaults) ---
def preprocess_image_default(image_path):
    """Loads and preprocesses a single image using default settings."""
    try: image = Image.open(image_path).convert("RGBA")
    except Exception as e: raise gr.Error(f"无法打开图像: {Path(image_path).name}")

    if DEFAULT_BACKGROUND_CHOICE == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else: # "自动去背景"
        image = remove_background(image, rembg_session, force=True)

    image = do_resize_content(image, DEFAULT_FOREGROUND_RATIO)
    image = expand_to_square(image)
    # Use default background color
    image = add_background(image, DEFAULT_BACKGROUND_COLOR)
    return image.convert("RGB")

def gen_image_default(input_image, seed):
    """Generates 3D model using default scale and steps."""
    global pipeline, model, args
    if pipeline is None or model is None: raise gr.Error("模型或处理管道未初始化!")
    pipeline.set_seed(seed)
    try:
        # Use default scale and step
        rt_dict = pipeline(input_image, scale=DEFAULT_GUIDANCE_SCALE, step=DEFAULT_STEP)
        stage1_images = rt_dict["stage1_images"]
        stage2_images = rt_dict["stage2_images"]
        np_imgs = np.concatenate(stage1_images, 1)
        np_xyzs = np.concatenate(stage2_images, 1)
        glb_path, obj_path = generate3d(model, np_imgs, np_xyzs, args.device)
        return Image.fromarray(np_imgs), Image.fromarray(np_xyzs), glb_path, obj_path
    except Exception as e:
        print(f"Error during 3D generation pipeline: {e}")
        raise gr.Error(f"生成3D模型时出错: {e}")

# --- Argument Parsing and Model Loading ---
# ... (no changes needed here, still loads models/configs) ...
parser = argparse.ArgumentParser()
parser.add_argument("--stage1_config", type=str, default="configs/nf7_v3_SNR_rd_size_stroke.yaml")
parser.add_argument("--stage2_config", type=str, default="configs/stage2-v2-snr.yaml")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
crm_path = "/mnt/ssd/fyz/CRM/CRM.pth"
xyz_path = "/mnt/ssd/fyz/CRM/ccm-diffusion.pth"
pixel_path = "/mnt/ssd/fyz/CRM/pixel-diffusion.pth"
css_path = "/home/fyz/CRM/styles.css"
examples_dir = "/home/fyz/CRM/restruct_examples"
if not Path(crm_path).exists(): raise FileNotFoundError(f"CRM model not found: {crm_path}")
if not Path(xyz_path).exists(): raise FileNotFoundError(f"XYZ Diffusion model not found: {xyz_path}")
if not Path(pixel_path).exists(): raise FileNotFoundError(f"Pixel Diffusion model not found: {pixel_path}")
if not Path(css_path).exists():
    print(f"Warning: CSS file not found at {css_path}. Using default styles.")
    css_content = ""
else:
    with open(css_path, 'r', encoding='utf-8') as file: css_content = file.read()
if not Path(examples_dir).is_dir(): raise NotADirectoryError(f"Examples directory not found: {examples_dir}")
specs = json.load(open("configs/specs_objaverse_total.json"))
model = CRM(specs).to(args.device)
model.load_state_dict(torch.load(crm_path, map_location=args.device), strict=False)
model.eval()
stage1_config = OmegaConf.load(args.stage1_config).config
stage2_config = OmegaConf.load(args.stage2_config).config
stage1_sampler_config = stage1_config.sampler
stage2_sampler_config = stage2_config.sampler
stage1_model_config = stage1_config.models
stage2_model_config = stage2_config.models
stage1_model_config.resume = pixel_path
stage2_model_config.resume = xyz_path
try:
    pipeline = TwoStagePipeline(
        stage1_model_config, stage2_model_config, stage1_sampler_config, stage2_sampler_config,
        device=args.device, dtype=torch.float16 if args.device == 'cuda' else torch.float32
    )
    print("Pipeline initialized successfully.")
except Exception as e:
    print(f"FATAL: Error initializing pipeline: {e}")
    exit()

prev_url = "http://localhost:8003"
next_url = "http://localhost:8005"

# --- Load ALL image paths and prepare initial gallery ---
all_image_paths = sorted([
    os.path.join(examples_dir, img)
    for img in os.listdir(examples_dir)
    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
])
total_images = len(all_image_paths)
initial_gallery_paths = all_image_paths[:BATCH_SIZE]
print(f"Found {total_images} images in {examples_dir}. Displaying first {len(initial_gallery_paths)}.")

# --- Gradio Interface ---
with gr.Blocks(css=css_content, theme="default", elem_classes="gradio-app") as demo:
    # --- Header and Titles ---
    # ... (no changes) ...
    with gr.Row(variant="panel", elem_classes="navbar"): gr.Markdown("Elongevity AI Lab", elem_classes="lab-title")
    with gr.Row(variant="panel", elem_classes="header"):
        with gr.Column(scale=1, elem_classes="nav-button-left"): gr.HTML(f'<a href="{prev_url}" class="nav-button" style="text-decoration: none;" target="_self">上一页</a>')
        with gr.Column(scale=2, elem_classes="title-container"): gr.Markdown("## 空间智能 / 生成", elem_classes="section-title")
        with gr.Column(scale=1, elem_classes="nav-button-right"): gr.HTML(f'<a href="{next_url}" class="nav-button" style="text-decoration: none;" target="_self">下一页</a>')
    with gr.Column(variant="panel"):
        gr.Markdown("# 实验室3D资源数据工厂", elem_classes="center-title")
        gr.Markdown("## 7-24 无间断生成数据", elem_classes="center-title")
    # Removed the extra "技术应用/模型输入" titles as input section is simpler now

    # --- Input Section (SIMPLIFIED LAYOUT) ---
    with gr.Row(variant="panel", elem_classes="input-container"):
        # Left Column: Gallery
        with gr.Column(scale=3): # Gallery slightly wider
            gr.Markdown("### 当前处理批次图像")
            examples_gallery = gr.Gallery(
                value=initial_gallery_paths,
                label="当前批次源图像", columns=5, height="auto", object_fit="contain",
                preview=True, interactive=False
            )
        # Right Column: Preview Only
        with gr.Column(scale=2):
            gr.Markdown("### 当前处理图像预览")
            current_image_display = gr.Image(
                label="当前处理图像预览", type="pil", image_mode="RGBA",
                # Adjust height if needed based on gallery height
                height=500, # Example height
                interactive=False, elem_classes="display-container"
            )
            # Parameter controls removed

    # --- Progress Reporting (Scrolling Log) ---
    with gr.Row(variant="panel"):
        progress_text = gr.Textbox(
            label="处理日志", value="等待开始...\n", interactive=False,
            lines=LOG_TEXTBOX_LINES, max_lines=LOG_TEXTBOX_LINES,
            autoscroll=True, elem_classes="progress-log"
        )

    # --- Control Buttons ---
    with gr.Row(variant="panel"):
        batch_button = gr.Button("开始批量生成 (分批处理)", variant="primary", elem_classes="button")
        stop_button = gr.Button("中止处理", variant="stop", interactive=False, elem_classes="button")


    # --- Output Section (REVISED LAYOUT V2) ---
    with gr.Row(variant="panel", elem_classes="header"):
         gr.Markdown("## 模型输出", elem_classes="center-title")

    with gr.Row(variant="panel", elem_classes="output-container"): # Main output row
        # Left Column: Stacked Images
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 视图输出")
            processed_image = gr.Image(
                 label="预处理后图像", interactive=False, type="pil",
                 image_mode="RGB", height=200, # Adjusted height
                 elem_classes="output-container-inner"
            )
            image_output = gr.Image(
                 interactive=False, label="多视角RGB图像",
                 height=150, # Adjusted height
                 elem_classes="output-container-inner"
            )
            xyz_output = gr.Image(
                interactive=False, label="多视角归一化坐标",
                height=150, # Adjusted height
                elem_classes="output-container-inner"
            )
            # Total height approx 200+150+150 = 500

        # Right Column: 3D Model and Download Area
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 3D模型")
            output_model = gr.Model3D(
                label="GLB模型预览", interactive=False,
                height=450, # Adjusted height to be slightly less than left column total
                elem_classes="model3d-output"
            )
            # Container for the download button, initially hidden
            with gr.Group(visible=False) as output_files_group:
                 batch_output = gr.File(
                    label="批次结果 Zip 下载", file_count="multiple",
                    file_types=[".zip"], interactive=False, # Will be set by generator
                    elem_classes="secondary small-download-box"
                 )

    # --- Footer ---
    with gr.Column():
        gr.HTML("<hr>")
        gr.Markdown("Elongevity AI Lab ©️ 2025", elem_classes="footer-text")

    # --- State and Logic ---
    processing_flag = threading.Event()
    current_image_index_state = gr.State(0)
    all_image_paths_state = gr.State(all_image_paths)
    saved_zip_paths_state = gr.State([])

    def append_log(current_log, message):
        return current_log + message + "\n"

    # --- Batch Process Generator (Simplified Inputs) ---
    def batch_process_generator(current_start_index, all_paths, saved_zip_paths_list,
                                current_log_text): # Removed parameter inputs

        log = current_log_text
        if not all_paths:
             log = append_log(log, "错误：未在指定目录找到图像文件。")
             yield {
                progress_text: gr.update(value=log),
                batch_button: gr.update(interactive=True), stop_button: gr.update(interactive=False),
                saved_zip_paths_state: [], output_files_group: gr.update(visible=False) # Ensure hidden on error
             }
             processing_flag.clear()
             return

        current_run_saved_zips = saved_zip_paths_list
        total_images_overall = len(all_paths)
        total_batches = math.ceil(total_images_overall / BATCH_SIZE)
        # Use default base seed
        seed_counter = DEFAULT_SEED + current_start_index

        # --- Batch Loop ---
        while current_start_index < total_images_overall and processing_flag.is_set():
            batch_num = (current_start_index // BATCH_SIZE) + 1
            batch_start_time = time.time()
            start_index = current_start_index
            end_index = min(start_index + BATCH_SIZE, total_images_overall)
            current_batch_paths = all_paths[start_index:end_index]
            current_batch_size = len(current_batch_paths)
            batch_info_str = f"批次 {batch_num}/{total_batches} ({start_index+1}-{end_index}/{total_images_overall})"
            print(f"\n--- Starting {batch_info_str} ---")

            log = append_log(log, f"--- {batch_info_str}: 准备中...")
            yield {
                examples_gallery: gr.update(value=current_batch_paths),
                progress_text: gr.update(value=log),
                current_image_display: gr.update(value=None),
                processed_image: gr.update(value=None), image_output: gr.update(value=None),
                xyz_output: gr.update(value=None), output_model: gr.update(value=None),
                saved_zip_paths_state: current_run_saved_zips,
                # Keep file group visibility based on whether zips exist
                output_files_group: gr.update(visible=bool(current_run_saved_zips))
            }

            batch_results_paths = []
            with tempfile.TemporaryDirectory() as batch_temp_dir:
                batch_temp_dir_path = Path(batch_temp_dir)

                # --- Image Loop within Batch ---
                for i, img_path in enumerate(current_batch_paths):
                    if not processing_flag.is_set():
                        log = append_log(log, f"{batch_info_str}: 处理中止于图像 {i+1}/{current_batch_size}")
                        yield { progress_text: gr.update(value=log) }
                        break

                    img_start_time = time.time()
                    current_img_index_overall = start_index + i
                    img_seed = seed_counter + i # Increment seed per image
                    base_name = Path(img_path).name
                    img_progress_prefix = f"{batch_info_str} ({i+1}/{current_batch_size}, {base_name})"

                    try:
                        current_img_pil = Image.open(img_path).convert("RGBA")
                        yield { current_image_display: gr.update(value=current_img_pil) }
                    except Exception as e:
                        print(f"{img_progress_prefix}: Error loading image for display: {e}")
                        log = append_log(log, f"{img_progress_prefix}: 错误 - 无法加载预览")
                        yield { progress_text: gr.update(value=log) }
                        continue

                    try:
                        log = append_log(log, f"{img_progress_prefix}: 1/4 图像预处理...")
                        yield { progress_text: gr.update(value=log) }
                        # Use default preprocess function
                        preprocessed_img = preprocess_image_default(img_path)
                        yield { processed_image: gr.update(value=preprocessed_img) }

                        log = append_log(log, f"{img_progress_prefix}: 2/4 生成多视角图...")
                        yield { progress_text: gr.update(value=log) }
                        # Use default gen_image function (pass only image and seed)
                        stage1_img, stage2_img, temp_glb_path_orig, temp_obj_path_orig = gen_image_default(preprocessed_img, img_seed)
                        yield { image_output: gr.update(value=stage1_img), xyz_output: gr.update(value=stage2_img) }

                        log = append_log(log, f"{img_progress_prefix}: 3/4 重建3D模型...")
                        yield { progress_text: gr.update(value=log) }

                        log = append_log(log, f"{img_progress_prefix}: 4/4 保存结果...")
                        yield { progress_text: gr.update(value=log) }
                        batch_temp_glb_path = batch_temp_dir_path / f"{Path(temp_glb_path_orig).name}"
                        batch_temp_obj_path = batch_temp_dir_path / f"{Path(temp_obj_path_orig).name}"
                        shutil.copy(temp_glb_path_orig, batch_temp_glb_path)
                        shutil.copy(temp_obj_path_orig, batch_temp_obj_path)
                        batch_results_paths.append((str(batch_temp_glb_path), str(batch_temp_obj_path)))

                        img_end_time = time.time()
                        log = append_log(log, f"{img_progress_prefix}: ✔️ 完成 [{(img_end_time-img_start_time):.1f}s]")
                        yield {
                            output_model: gr.update(value=str(batch_temp_glb_path)),
                            progress_text: gr.update(value=log)
                        }

                    except Exception as e:
                        img_end_time = time.time()
                        print(f"!!! {img_progress_prefix}: Error during processing: {e}")
                        log = append_log(log, f"{img_progress_prefix}: ❌ 错误 - 跳过 ({e})")
                        yield {
                            progress_text: gr.update(value=log),
                            processed_image: gr.update(value=None), image_output: gr.update(value=None),
                            xyz_output: gr.update(value=None), output_model: gr.update(value=None),
                        }
                        continue
                # --- End of Image Loop ---

                if batch_results_paths:
                    completed_in_batch = len(batch_results_paths)
                    batch_zip_filename = f"batch_{batch_num:03d}_results_{start_index+1}-{start_index+completed_in_batch}.zip"
                    batch_zip_path_temp = batch_temp_dir_path / batch_zip_filename
                    log = append_log(log, f"{batch_info_str}: 压缩 {completed_in_batch} 个结果...")
                    yield { progress_text: gr.update(value=log) }

                    # ... (Zipping and Saving logic remains the same) ...
                    print(f"Creating zip file for Batch {batch_num}: {batch_zip_path_temp}")
                    with zipfile.ZipFile(batch_zip_path_temp, 'w') as zipf:
                        for glb, obj in batch_results_paths:
                            zipf.write(glb, Path(glb).name)
                            zipf.write(obj, Path(obj).name)
                    print("Zip file created.")
                    save_path = RESULTS_SAVE_DIR / batch_zip_filename
                    try:
                        shutil.copy(batch_zip_path_temp, save_path)
                        print(f"Batch {batch_num} Zip successfully saved to: {save_path}")
                        save_message = f"Zip包已保存至 {RESULTS_SAVE_DIR.name}/{save_path.name}"
                        current_run_saved_zips.append(str(save_path))
                    except Exception as e:
                        print(f"Error saving Batch {batch_num} zip to {RESULTS_SAVE_DIR}: {e}")
                        save_message = f"Zip包本地保存失败! ({e})"

                    batch_end_time = time.time()
                    log = append_log(log, f"{batch_info_str}: ✔️ 批次完成 ({completed_in_batch}/{current_batch_size} 模型). {save_message} [耗时: {(batch_end_time-batch_start_time):.1f}s]")
                    yield {
                        batch_output: gr.update(value=current_run_saved_zips, interactive=True),
                        progress_text: gr.update(value=log),
                        saved_zip_paths_state: current_run_saved_zips,
                        output_files_group: gr.update(visible=True) # Make group visible
                    }
                else:
                    batch_end_time = time.time()
                    print(f"No results generated for Batch {batch_num}, skipping zip file.")
                    log = append_log(log, f"{batch_info_str}: ✔️ 批次完成. 未生成模型. [耗时: {(batch_end_time-batch_start_time):.1f}s]")
                    yield { progress_text: gr.update(value=log) } # No change to zip list/state/visibility

            if not processing_flag.is_set():
                 log = append_log(log, f"--- 用户请求停止 ---")
                 yield { progress_text: gr.update(value=log) }
                 print("Stop requested after finishing batch zip.")
                 break

            current_start_index = end_index
        # --- End of Batch Loop ---

        final_status = "全部处理完成" if processing_flag.is_set() else "处理中止"
        print(f"All batch processing ended. Status: {final_status}.")
        final_progress_text = f"{final_status}. 共处理 {current_start_index}/{total_images_overall} 张图像。"
        log = append_log(log, f"--- {final_progress_text} ---")
        yield {
            progress_text: gr.update(value=log),
            saved_zip_paths_state: current_run_saved_zips,
            # Keep file group visible if any zips were ever generated
            output_files_group: gr.update(visible=bool(current_run_saved_zips))
        }

    # --- Control Functions (Simplified) ---
    def start_batch_processing():
        print("Start batch processing requested.")
        processing_flag.set()
        initial_log = "处理已开始 (使用默认参数)...\n"
        return {
            current_image_index_state: 0,
            saved_zip_paths_state: [],
            batch_output: gr.update(value=[], interactive=False),
            output_files_group: gr.update(visible=False), # Hide download group initially
            batch_button: gr.update(interactive=False),
            stop_button: gr.update(interactive=True),
            progress_text: gr.update(value=initial_log)
        }

    def stop_processing_action(current_log_text):
        print("Stop processing requested.")
        processing_flag.clear()
        log = append_log(current_log_text, "--- 中止指令已发送... 等待当前步骤完成... ---")
        return {
             progress_text: gr.update(value=log),
             stop_button: gr.update(interactive=False)
        }

    def finish_processing_updates(current_log_text):
        print("Finishing processing updates.")
        processing_flag.clear()
        log = append_log(current_log_text, "--- 处理流程结束 ---")
        # Visibility of download group determined by final generator yield
        return {
            batch_button: gr.update(interactive=True),
            stop_button: gr.update(interactive=False),
            progress_text: gr.update(value=log)
        }

    # --- Event Listeners (Simplified Inputs) ---
    batch_process_event = batch_button.click(
        fn=start_batch_processing,
        # Reset states and controls
        outputs=[current_image_index_state, saved_zip_paths_state, batch_output,
                 output_files_group, # Add group to outputs
                 batch_button, stop_button, progress_text]
    ).then(
        fn=batch_process_generator,
        inputs=[ # Removed parameter inputs
            current_image_index_state, all_image_paths_state, saved_zip_paths_state,
            progress_text,
        ],
        outputs=[ # Added output_files_group
            examples_gallery, progress_text,
            batch_output, current_image_display, processed_image,
            image_output, xyz_output, output_model,
            saved_zip_paths_state,
            output_files_group # Update visibility
        ]
    ).then(
       fn=finish_processing_updates,
       inputs=[progress_text],
       outputs=[batch_button, stop_button, progress_text] # Don't need to update file group visibility here
    )

    stop_button.click(
        fn=stop_processing_action,
        inputs=[progress_text],
        outputs=[progress_text, stop_button],
        cancels=[batch_process_event]
    )

# --- Launch App ---
if __name__ == "__main__":
    if not all_image_paths:
        print("\nWARNING: No images found in the specified directory.")
        print(f"Please check the directory: {examples_dir}\n")

    demo.queue().launch(server_name="0.0.0.0", server_port=8004, share=False)