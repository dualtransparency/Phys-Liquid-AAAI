import os
import asyncio
import io
import json
import logging
import tempfile
import zipfile
import httpx # Import the HTTP client library

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware # Ensure this is imported
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import rembg
# Assuming these local imports exist and are correct relative to http_server.py
from model import CRM
from pipelines import TwoStagePipeline
from inference import generate3d # <--- MAKE SURE THIS IMPORT IS CORRECT
from libs.base_utils import do_resize_content # Needed for preprocess_image

# --- Configuration & Constants ---
# Added format for clarity, ensure logs show timestamps etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default AI parameters (Hardcoded)
DEFAULT_SCALE = 5.0
DEFAULT_STEP = 50
DEFAULT_BG_CHOICE = "Auto Remove background"
DEFAULT_FOREGROUND_RATIO = 1.0
DEFAULT_BACKGROUND_COLOR = (127, 127, 127) # RGB tuple for the added background
OUTPUT_PIXEL_PNG = "pixel_images.png"
OUTPUT_XYZ_PNG = "xyz_images.png"
OUTPUT_3D_ZIP = "output3d.zip" # Name for the callback zip file

# Paths to models (Update these paths as needed)
# Using environment variables or a config file is generally better, but keeping as is per original code
CRM_MODEL_PATH = os.getenv("CRM_MODEL_PATH", "/mnt/ssd/fyz/CRM/CRM.pth")
PIXEL_DIFFUSION_PATH = os.getenv("PIXEL_DIFFUSION_PATH", "/mnt/ssd/fyz/CRM/pixel-diffusion.pth")
XYZ_DIFFUSION_PATH = os.getenv("XYZ_DIFFUSION_PATH", "/mnt/ssd/fyz/CRM/ccm-diffusion.pth")
SPECS_PATH = os.getenv("SPECS_PATH", "configs/specs_objaverse_total.json")
STAGE1_CONFIG_PATH = os.getenv("STAGE1_CONFIG_PATH", "configs/nf7_v3_SNR_rd_size_stroke.yaml")
STAGE2_CONFIG_PATH = os.getenv("STAGE2_CONFIG_PATH", "configs/stage2-v2-snr.yaml")

# Callback Configuration
CALLBACK_TIMEOUT = 60.0 # Timeout in seconds for callback requests

# --- Global Variables ---
# Added title/version for documentation purposes
app = FastAPI(title="CRM 3D Reconstruction Service", version="1.1.0") # Updated version
gpu_lock = asyncio.Lock() # Lock remains crucial for GPU resource serialization
models_loaded = False
crm_model = None
pipeline = None
rembg_session = None
# Global httpx client session for potential connection pooling/reuse
http_client: httpx.AsyncClient = None

# --- Add CORS Middleware ---
# Allow all origins for simplicity, adjust in production for security
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Be careful with this in production if origins is "*"
    allow_methods=["*"],    # Allows all standard methods
    allow_headers=["*"],    # Allows all standard headers
)

# --- Helper Functions --- (Keep the image processing functions as they were in the previous version)
def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    """Expands image to square shape with specified background color (defaults to transparent)."""
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    if len(bg_color) == 4 and bg_color[3] < 255:
        new_image = Image.new("RGBA", new_size, bg_color)
        paste_mode = "RGBA"
    else:
        new_image = Image.new("RGB", new_size, bg_color[:3])
        paste_mode = "RGB"

    if image.mode != paste_mode and image.mode != "RGBA":
         image = image.convert(paste_mode if paste_mode=="RGB" else "RGBA")
    elif image.mode == "RGBA" and paste_mode == "RGB":
         new_image = Image.new("RGBA", new_size, bg_color[:3] + (255,))
         paste_mode = "RGBA"

    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)

    if paste_mode == "RGBA" and image.mode == "RGBA":
         new_image.paste(image, paste_position, image)
    else:
         new_image.paste(image, paste_position)

    return new_image

def remove_background(
        image: Image.Image,
        rembg_session_local=None,
        force: bool = False,
        **rembg_kwargs,
) -> Image.Image:
    """Removes background using rembg library, ensuring RGBA output."""
    original_mode = image.mode
    if image.mode == "RGBA":
        try:
            alpha_bbox = image.getchannel('A').getbbox()
            if alpha_bbox is None and not force:
                 logger.info("Image is already fully transparent, skipping rembg.")
                 return image
            min_alpha, max_alpha = image.getchannel('A').getextrema()
            if min_alpha < 255 and max_alpha > 0 and not force:
                logger.info("Alpha channel is partially transparent, skipping rembg unless forced.")
                return image
        except Exception as e:
            logger.warning(f"Could not efficiently check alpha channel: {e}. Proceeding with rembg check.")

    logger.info(f"Applying rembg background removal (force={force})...")
    try:
        if image.mode not in ["RGB", "RGBA"]:
            image_for_rembg = image.convert("RGBA")
        else:
            image_for_rembg = image
        image_out = rembg.remove(image_for_rembg, session=rembg_session_local, **rembg_kwargs)
        if image_out.mode != "RGBA":
             logger.warning("Rembg did not return RGBA, converting.")
             image_out = image_out.convert("RGBA")
        return image_out
    except Exception as e:
        logger.error(f"Rembg failed: {e}", exc_info=True)
        logger.warning("Falling back to original image due to rembg error.")
        if original_mode != "RGBA":
             return image.convert("RGBA")
        else:
             return image

def add_background(image, bg_color=(255, 255, 255)):
    """Adds a solid RGB background color to an RGBA image, returning RGBA."""
    if image.mode != "RGBA":
        logger.warning("Image mode is not RGBA before adding background. Converting.")
        image = image.convert("RGBA")
    if len(bg_color) != 3:
        logger.error(f"Invalid background color format: {bg_color}. Using default white.")
        bg_color = (255, 255, 255)

    background = Image.new("RGBA", image.size, bg_color + (255,))
    final_image = Image.alpha_composite(background, image)
    return final_image

def preprocess_image(image: Image.Image, background_choice: str, foreground_ratio: float, background_color: tuple, rembg_session_local):
    """Preprocesses the input image for the AI model."""
    logger.info(f"Preprocessing image with bg_choice: {background_choice}, ratio: {foreground_ratio}")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if background_choice == "Alpha as mask":
        logger.info("Using alpha channel as mask (no explicit removal).")
    elif background_choice == "Auto Remove background":
        logger.info("Attempting automatic background removal.")
        image = remove_background(image, rembg_session_local=rembg_session_local, force=True)
    else:
        logger.warning(f"Unknown background_choice '{background_choice}', defaulting to Auto Remove.")
        image = remove_background(image, rembg_session_local=rembg_session_local, force=True)
    if foreground_ratio < 0.99 or foreground_ratio > 1.01:
        logger.info(f"Resizing content with scale_rate: {foreground_ratio}")
        image = do_resize_content(image, foreground_ratio)
    logger.info("Expanding image to square.")
    image = expand_to_square(image, bg_color=(0, 0, 0, 0))
    logger.info(f"Adding solid background color: {background_color}")
    image = add_background(image, background_color)
    logger.info("Converting final image to RGB.")
    return image.convert("RGB")

# --- Callback Helper Functions --- (Keep as they were)
async def send_result_callback(client: httpx.AsyncClient, callback_url: str, task_id: str, file_name: str, content_type: str, file_bytes: bytes):
    """Sends a result file part via HTTP POST callback."""
    if not client:
        logger.error(f"Callback failed for {file_name} (Task: {task_id}): HTTP client not initialized.")
        return False
    if not callback_url:
        logger.error(f"Callback failed for {file_name} (Task: {task_id}): No callback URL provided.")
        return False

    target_url = f"{callback_url.rstrip('/')}/result/{task_id}"
    files = {'file': (file_name, file_bytes, content_type)}
    data = {'name': file_name}
    try:
        logger.info(f"Sending callback for {file_name} (Task: {task_id}) to {target_url}...")
        response = await client.post(target_url, files=files, data=data, timeout=CALLBACK_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Successfully sent callback for {file_name} (Task: {task_id}) to {target_url}. Status: {response.status_code}")
        return True
    except httpx.TimeoutException:
        logger.error(f"Callback request timed out for {file_name} (Task: {task_id}) to {target_url} after {CALLBACK_TIMEOUT}s.")
    except httpx.RequestError as e:
        logger.error(f"Callback request failed for {file_name} (Task: {task_id}) to {target_url}: Network error - {type(e).__name__}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Callback request failed for {file_name} (Task: {task_id}) to {target_url}: Status {e.response.status_code}, Response: {e.response.text[:500]}...")
    except Exception as e:
        logger.error(f"Unexpected error during callback for {file_name} (Task: {task_id}) to {target_url}: {e}", exc_info=True)
    return False

async def send_status_callback(client: httpx.AsyncClient, callback_url: str, task_id: str, status: str, error_message: str = None):
    """Sends the final status (completed/failed) via HTTP POST callback."""
    if not client:
        logger.error(f"Status callback failed (Task: {task_id}, Status: {status}): HTTP client not initialized.")
        return False
    if not callback_url:
        logger.error(f"Status callback failed (Task: {task_id}, Status: {status}): No callback URL provided.")
        return False

    target_url = f"{callback_url.rstrip('/')}/status"
    payload = {"taskId": task_id, "status": status}
    if error_message:
        payload["error"] = error_message
    try:
        logger.info(f"Sending status '{status}' callback (Task: {task_id}) to {target_url}...")
        response = await client.post(target_url, json=payload, timeout=CALLBACK_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Successfully sent status '{status}' callback (Task: {task_id}) to {target_url}. Status: {response.status_code}")
        return True
    except httpx.TimeoutException:
         logger.error(f"Status callback request timed out (Task: {task_id}, Status: {status}) to {target_url} after {CALLBACK_TIMEOUT}s.")
    except httpx.RequestError as e:
        logger.error(f"Status callback request failed (Task: {task_id}, Status: {status}) to {target_url}: Network error - {type(e).__name__}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Status callback request failed (Task: {task_id}, Status: {status}) to {target_url}: Status {e.response.status_code}, Response: {e.response.text[:500]}...")
    except Exception as e:
        logger.error(f"Unexpected error during status callback (Task: {task_id}, Status: {status}) to {target_url}: {e}", exc_info=True)
    return False

# --- Model Loading Function --- (Keep as it was)
def load_models():
    """Loads all AI models and necessary components."""
    global models_loaded, crm_model, pipeline, rembg_session
    models_loaded = False
    try:
        logger.info("Loading Rembg session...")
        rembg_session = rembg.new_session()
        logger.info("Rembg session loaded.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This service requires a GPU.")
        if torch.cuda.device_count() == 0:
             raise RuntimeError("CUDA available but no CUDA devices found.")
        device = "cuda"
        logger.info(f"Using device: {device} (Found {torch.cuda.device_count()} CUDA devices)")
        logger.info(f"Loading CRM model from {CRM_MODEL_PATH}...")
        if not os.path.exists(CRM_MODEL_PATH):
            raise FileNotFoundError(f"CRM model not found at {CRM_MODEL_PATH}")
        if not os.path.exists(SPECS_PATH):
            raise FileNotFoundError(f"Specs file not found at {SPECS_PATH}")
        with open(SPECS_PATH, 'r') as f:
            specs = json.load(f)
        crm_model = CRM(specs).to(device)
        logger.info("Loading CRM state dictionary...")
        state_dict = torch.load(CRM_MODEL_PATH, map_location=device)
        crm_model.load_state_dict(state_dict, strict=False)
        crm_model.eval()
        logger.info("CRM model loaded successfully.")
        logger.info("Loading Pipeline models...")
        if not os.path.exists(STAGE1_CONFIG_PATH):
             raise FileNotFoundError(f"Stage 1 config not found: {STAGE1_CONFIG_PATH}")
        if not os.path.exists(STAGE2_CONFIG_PATH):
             raise FileNotFoundError(f"Stage 2 config not found: {STAGE2_CONFIG_PATH}")
        if not os.path.exists(PIXEL_DIFFUSION_PATH):
            raise FileNotFoundError(f"Pixel diffusion model not found: {PIXEL_DIFFUSION_PATH}")
        if not os.path.exists(XYZ_DIFFUSION_PATH):
            raise FileNotFoundError(f"XYZ diffusion model not found: {XYZ_DIFFUSION_PATH}")
        logger.info("Loading OmegaConf configurations...")
        stage1_config = OmegaConf.load(STAGE1_CONFIG_PATH)
        stage2_config = OmegaConf.load(STAGE2_CONFIG_PATH)
        if 'config' in stage1_config: stage1_config = stage1_config.config
        if 'config' in stage2_config: stage2_config = stage2_config.config
        try:
            OmegaConf.update(stage1_config, "models.resume", PIXEL_DIFFUSION_PATH, merge=True)
            OmegaConf.update(stage2_config, "models.resume", XYZ_DIFFUSION_PATH, merge=True)
            OmegaConf.update(stage1_config, "models.device", device, merge=True)
            OmegaConf.update(stage2_config, "models.device", device, merge=True)
            logger.info("Updated model resume paths and device in configs.")
        except Exception as e:
             logger.warning(f"Could not automatically update resume paths/device in OmegaConf objects: {e}. Ensure paths/device are correct in YAML files or pipeline handles this.")
        logger.info("Initializing TwoStagePipeline...")
        pipeline_init_args = {
            "stage1_model_config": stage1_config.models,
            "stage2_model_config": stage2_config.models,
            "stage1_sampler_config": stage1_config.sampler,
            "stage2_sampler_config": stage2_config.sampler,
        }
        pipeline = TwoStagePipeline(**pipeline_init_args)
        logger.info("Pipeline loaded successfully.")
        models_loaded = True
        logger.info("All models loaded successfully and ready on device '%s'.", device)
    except FileNotFoundError as e:
        logger.error(f"Model or config file not found: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error during model loading (possibly CUDA/GPU issue): {e}", exc_info=True)
    except ImportError as e:
        logger.error(f"Import error during model loading (check dependencies): {e}", exc_info=True)
    except OmegaConf.errors.ConfigKeyError as e:
         logger.error(f"Configuration key error (check YAML structure/paths): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred loading models: {e}", exc_info=True)
    if not models_loaded:
        logger.critical("MODEL LOADING FAILED.")

# --- FastAPI Startup/Shutdown Events --- (Keep as they were)
@app.on_event("startup")
async def startup_event():
    global http_client
    logger.info("Application startup sequence initiated...")
    http_client = httpx.AsyncClient(timeout=CALLBACK_TIMEOUT + 10.0)
    logger.info(f"HTTP client initialized with timeout {CALLBACK_TIMEOUT + 10.0}s.")
    logger.info("Submitting model loading task to background executor...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_models)
    if models_loaded:
        logger.info("Model loading process completed successfully.")
    else:
        logger.critical("Model loading process failed. Service may be unavailable or unstable.")
    logger.info("Application startup sequence finished.")

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    logger.info("Application shutdown sequence initiated...")
    if http_client:
        logger.info("Closing HTTP client...")
        await http_client.aclose()
        logger.info("HTTP client closed.")
    else:
        logger.info("HTTP client was not initialized or already closed.")
    logger.info("Application shutdown sequence finished.")

# --- Background Processing Task ---
async def process_reconstruction_task(
    image_bytes: bytes,
    task_id: str,
    callback_url: str,
    client: httpx.AsyncClient
    ):
    """The core AI processing logic, designed to run in the background."""
    logger.info(f"[Task {task_id}] Starting background processing...")
    # Paths for files created by generate3d (using tempfile.NamedTemporaryFile with delete=False)
    glb_path_returned = None
    zip_path_returned = None
    obj_path_generated = None # Track associated files for cleanup
    mtl_path_generated = None
    png_path_generated = None

    processing_status = "failed"
    error_message = "Unknown processing error"
    results_sent = {"pixel": False, "xyz": False, "zip": False}

    try:
        if not models_loaded:
             logger.error(f"[Task {task_id}] Models are not loaded. Aborting task.")
             raise RuntimeError("Models are not loaded, cannot process task.")

        logger.info(f"[Task {task_id}] Waiting to acquire GPU lock...")
        async with gpu_lock:
            logger.info(f"[Task {task_id}] Acquired GPU lock.")
            try:
                # 1. Load image
                try:
                    input_image = Image.open(io.BytesIO(image_bytes))
                    input_image = input_image.convert("RGBA")
                    logger.info(f"[Task {task_id}] Input image loaded ({input_image.mode}, {input_image.size}).")
                except Exception as e:
                    logger.error(f"[Task {task_id}] Failed to decode image: {e}", exc_info=True)
                    raise ValueError("Invalid image data provided.")

                # 2. Preprocess image
                logger.info(f"[Task {task_id}] Preprocessing image...")
                preprocessed_image = preprocess_image(
                    input_image, DEFAULT_BG_CHOICE, DEFAULT_FOREGROUND_RATIO,
                    DEFAULT_BACKGROUND_COLOR, rembg_session
                )
                logger.info(f"[Task {task_id}] Image preprocessing completed.")

                # 3. Run pipeline
                logger.info(f"[Task {task_id}] Running Stage 1 & 2 pipeline...")
                with torch.no_grad():
                    rt_dict = pipeline(preprocessed_image, scale=DEFAULT_SCALE, step=DEFAULT_STEP)
                if "stage1_images" not in rt_dict or "stage2_images" not in rt_dict:
                    logger.error(f"[Task {task_id}] Pipeline output missing expected keys.")
                    raise RuntimeError("Pipeline output structure incorrect.")
                np_imgs_combined = np.concatenate(rt_dict["stage1_images"], axis=1)
                np_xyzs_combined = np.concatenate(rt_dict["stage2_images"], axis=1)
                logger.info(f"[Task {task_id}] Pipeline completed. Combined shapes: pixel={np_imgs_combined.shape}, xyz={np_xyzs_combined.shape}")

                # 4. Send Pixel Images PNG
                pixel_png_buffer = io.BytesIO()
                Image.fromarray(np_imgs_combined).save(pixel_png_buffer, format="PNG")
                pixel_png_buffer.seek(0)
                if await send_result_callback(client, callback_url, task_id, OUTPUT_PIXEL_PNG, "image/png", pixel_png_buffer.getvalue()):
                    results_sent["pixel"] = True
                else:
                    logger.warning(f"[Task {task_id}] Failed callback for {OUTPUT_PIXEL_PNG}. Continuing...")
                pixel_png_buffer.close()

                # 5. Send XYZ Images PNG
                xyz_png_buffer = io.BytesIO()
                Image.fromarray(np_xyzs_combined).save(xyz_png_buffer, format="PNG")
                xyz_png_buffer.seek(0)
                if await send_result_callback(client, callback_url, task_id, OUTPUT_XYZ_PNG, "image/png", xyz_png_buffer.getvalue()):
                     results_sent["xyz"] = True
                else:
                     logger.warning(f"[Task {task_id}] Failed callback for {OUTPUT_XYZ_PNG}. Continuing...")
                xyz_png_buffer.close()

                # 6. Generate 3D Output using generate3d from inference.py
                logger.info(f"[Task {task_id}] Generating 3D output...")
                with torch.no_grad():
                    # <<<< MODIFIED CALL >>>>
                    # Pass arguments positionally as defined in generate3d
                    glb_path_returned, zip_path_returned = generate3d(
                        crm_model,          # Argument 1: model
                        np_imgs_combined,   # Argument 2: rgb
                        np_xyzs_combined,   # Argument 3: ccm
                        "cuda"              # Argument 4: device
                    )
                    # <<<< END MODIFIED CALL >>>>

                # Store related paths for cleanup (assuming generate3d naming convention)
                if zip_path_returned:
                     base_name_zip = zip_path_returned.rsplit('.zip', 1)[0]
                     obj_path_generated = f"{base_name_zip}.obj"
                     mtl_path_generated = f"{base_name_zip}.mtl"
                     png_path_generated = f"{base_name_zip}.png"

                # Check if the *required* zip file exists
                if not zip_path_returned or not os.path.exists(zip_path_returned):
                    logger.error(f"[Task {task_id}] generate3d did not produce or return a valid zip file path. Expected path pattern might differ. Got: {zip_path_returned}")
                    raise RuntimeError("3D generation failed to produce output zip.")
                logger.info(f"[Task {task_id}] 3D generation completed. Output zip: {zip_path_returned}, GLB: {glb_path_returned}")

                # 7. Send 3D Output ZIP
                logger.info(f"[Task {task_id}] Reading and sending {OUTPUT_3D_ZIP} via callback from {zip_path_returned}...")
                try:
                    with open(zip_path_returned, "rb") as f:
                        zip_content = f.read()
                    # Use OUTPUT_3D_ZIP as the filename in the callback for consistency
                    if await send_result_callback(client, callback_url, task_id, OUTPUT_3D_ZIP, "application/zip", zip_content):
                        results_sent["zip"] = True
                        logger.info(f"[Task {task_id}] Successfully sent {OUTPUT_3D_ZIP} ({len(zip_content)} bytes).")
                    else:
                        raise RuntimeError(f"Failed to send final callback for {OUTPUT_3D_ZIP}")
                except FileNotFoundError:
                    logger.error(f"[Task {task_id}] Temp zip file not found during send: {zip_path_returned}")
                    raise RuntimeError("Failed to read generated 3D zip.")
                except Exception as e:
                    logger.error(f"[Task {task_id}] Error reading/sending zip file {zip_path_returned}: {e}", exc_info=True)
                    raise

                if results_sent["zip"]:
                    processing_status = "completed"
                    error_message = None
                    logger.info(f"[Task {task_id}] Processing successful.")
                else:
                    processing_status = "failed"
                    error_message = f"Failed to send final {OUTPUT_3D_ZIP} callback."
                    logger.error(f"[Task {task_id}] Processing failed: {error_message}")

            except (ValueError, RuntimeError, FileNotFoundError, TypeError) as e: # Added TypeError
                logger.error(f"[Task {task_id}] Processing error: {e}", exc_info=True)
                error_message = f"Processing error: {str(e)[:500]}"
                processing_status = "failed"
            except Exception as e:
                logger.error(f"[Task {task_id}] Unexpected processing error: {e}", exc_info=True)
                error_message = "Internal server error during processing."
                processing_status = "failed"
            finally:
                logger.info(f"[Task {task_id}] Releasing GPU lock.")

    except Exception as e:
        logger.error(f"[Task {task_id}] System error outside processing lock: {e}", exc_info=True)
        error_message = f"System error: {str(e)[:500]}"
        processing_status = "failed"

    finally:
        # --- Send Final Status Callback ---
        logger.info(f"[Task {task_id}] Sending final status '{processing_status}' callback...")
        if client and callback_url:
             await send_status_callback(client, callback_url, task_id, processing_status, error_message)
        else:
             logger.error(f"[Task {task_id}] Cannot send status callback: HTTP client or callback URL invalid.")

        # --- Cleanup Temporary Files CREATED BY generate3d ---
        # <<<< MODIFIED CLEANUP >>>>
        files_to_clean = []
        if zip_path_returned and os.path.exists(zip_path_returned): files_to_clean.append(zip_path_returned)
        if obj_path_generated and os.path.exists(obj_path_generated): files_to_clean.append(obj_path_generated)
        if mtl_path_generated and os.path.exists(mtl_path_generated): files_to_clean.append(mtl_path_generated)
        if png_path_generated and os.path.exists(png_path_generated): files_to_clean.append(png_path_generated)
        if glb_path_returned and os.path.exists(glb_path_returned): files_to_clean.append(glb_path_returned)

        if files_to_clean:
            logger.info(f"[Task {task_id}] Cleaning up temporary files created by generate3d: {files_to_clean}")
            for file_path in files_to_clean:
                 try:
                     os.remove(file_path)
                     logger.debug(f"[Task {task_id}] Removed temp file: {file_path}")
                 except OSError as e:
                     logger.error(f"[Task {task_id}] Error removing temporary file {file_path}: {e}")
        else:
            logger.info(f"[Task {task_id}] No temporary files from generate3d found to clean up (or paths were None).")
        # <<<< END MODIFIED CLEANUP >>>>

        logger.info(f"[Task {task_id}] Background processing finished. Final status: {processing_status}")


# --- HTTP Endpoint --- (Keep as it was)
@app.post("/generate3d", status_code=202, tags=["Generation"])
async def http_generate3d(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image file to reconstruct (PNG, JPG, etc.)."),
    task_id: str = Form(..., description="Unique identifier for this task provided by the client."),
    callback_url: str = Form(..., description="Base URL where results (POST to /result/{taskId}) and status (POST to /status) will be sent back.")
    ):
    """
    Accepts an image, task ID, and callback URL.

    Validates input and queues the 3D generation task to run in the background.
    Returns immediately with status 'accepted'. Results and final status
    are sent via HTTP POST callbacks to the provided callback URL's subpaths.
    """
    logger.info(f"Received task request: task_id='{task_id}', callback_url='{callback_url}', image='{image.filename}' ({image.content_type})")
    if not models_loaded:
        logger.error(f"Task {task_id}: Refused - Models not loaded.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Models not ready.")
    if not http_client:
         logger.error(f"Task {task_id}: Refused - HTTP client not initialized.")
         raise HTTPException(status_code=503, detail="Service Unavailable: Internal HTTP client not ready.")
    if not callback_url or not callback_url.startswith(("http://", "https://")):
        logger.error(f"Task {task_id}: Invalid callback_url: {callback_url}")
        raise HTTPException(status_code=400, detail="Invalid callback_url format. Must be a valid URL starting with http:// or https://")
    try:
        image_bytes = await image.read()
        if not image_bytes:
            logger.error(f"Task {task_id}: Received empty image file.")
            raise ValueError("Received empty image file.")
        logger.info(f"Task {task_id}: Read {len(image_bytes)} bytes from image '{image.filename}'.")
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to read image file: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")
    finally:
        await image.close()
    logger.info(f"Task {task_id}: Queuing reconstruction task...")
    try:
        background_tasks.add_task(
            process_reconstruction_task,
            image_bytes=image_bytes,
            task_id=task_id,
            callback_url=callback_url,
            client=http_client
            )
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to add task to background queue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to queue task for processing.")
    logger.info(f"Task {task_id} accepted and queued.")
    return {"status": "accepted", "task_id": task_id, "detail": "Task queued. Results via callback."}

# --- Health Check Endpoint --- (Keep as it was)
@app.get("/health", tags=["System"])
async def health_check():
    """
    Provides a health check of the service.

    Returns 'healthy' if models are loaded and the HTTP client is ready,
    otherwise 'unhealthy'. Use this to monitor service readiness.
    """
    is_healthy = models_loaded and bool(http_client)
    status = "healthy" if is_healthy else "unhealthy"
    health_details = {
        "status": status,
        "models_loaded": models_loaded,
        "http_client_ready": bool(http_client)
    }
    log_level = logging.INFO if is_healthy else logging.WARNING
    logger.log(log_level, f"Health check result: {health_details}")
    return health_details

# --- Main Execution --- (Keep as it was)
if __name__ == "__main__":
    server_host = os.getenv("HOST", "0.0.0.0")
    try:
        server_port = int(os.getenv("PORT", "8001"))
    except ValueError:
        logger.warning(f"Invalid PORT environment variable. Using default 8001.")
        server_port = 8001
    reload_flag = os.getenv("USE_RELOAD", "false").lower() in ["true", "1", "yes"]
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    logger.info(f"Starting Uvicorn server...")
    logger.info(f" Host: {server_host}")
    logger.info(f" Port: {server_port}")
    logger.info(f" Reload: {reload_flag}")
    logger.info(f" Log Level: {log_level}")
    uvicorn.run(
        "http_server:app",
        host=server_host,
        port=server_port,
        reload=reload_flag,
        log_level=log_level
    )