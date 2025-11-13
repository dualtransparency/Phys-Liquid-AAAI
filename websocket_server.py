import os
import asyncio
import io
import json
import logging
import tempfile
import zipfile
from contextlib import contextmanager
import shutil # Keep for potential future use, maybe less needed now

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import rembg
from model import CRM
from pipelines import TwoStagePipeline
from inference import generate3d  # Assuming this function exists as used in run.py
from libs.base_utils import do_resize_content  # Needed for preprocess_image

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default AI parameters (Hardcoded as per design doc)
DEFAULT_SCALE = 5.0
DEFAULT_STEP = 50
DEFAULT_BG_CHOICE = "Auto Remove background"
DEFAULT_FOREGROUND_RATIO = 1.0
DEFAULT_BACKGROUND_COLOR = (127, 127, 127)
OUTPUT_PIXEL_PNG = "pixel_images.png"
OUTPUT_XYZ_PNG = "xyz_images.png"
OUTPUT_3D_ZIP = "output3d.zip" # This is now just a name identifier for the client

# Paths to models (Update these paths as needed)
CRM_MODEL_PATH = "/mnt/ssd/fyz/CRM/CRM.pth"
PIXEL_DIFFUSION_PATH = "/mnt/ssd/fyz/CRM/pixel-diffusion.pth"
XYZ_DIFFUSION_PATH = "/mnt/ssd/fyz/CRM/ccm-diffusion.pth"
SPECS_PATH = "configs/specs_objaverse_total.json"
STAGE1_CONFIG_PATH = "configs/nf7_v3_SNR_rd_size_stroke.yaml"
STAGE2_CONFIG_PATH = "configs/stage2-v2-snr.yaml"

# --- Global Variables ---
app = FastAPI()
gpu_lock = asyncio.Lock()
models_loaded = False
crm_model = None
pipeline = None
rembg_session = None


# --- Helper Functions (Keep existing ones: expand_to_square, remove_background, add_background, preprocess_image) ---
# (Assuming the helper functions from the previous code are here)
def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def remove_background(
        image: Image.Image,
        rembg_session_local=None,
        force: bool = False,
        **rembg_kwargs,
) -> Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        logger.info("Alpha channel not empty, using alpha channel as mask (no bg removal)")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        try:
            image = rembg.remove(image, session=rembg_session_local, **rembg_kwargs)
        except Exception as e:
            logger.error(f"Rembg failed: {e}", exc_info=True)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def add_background(image, bg_color=(255, 255, 255)):
    background = Image.new("RGBA", image.size, bg_color)
    final_image = Image.alpha_composite(background, image)
    return final_image


def preprocess_image(image, background_choice, foreground_ratio, background_color, rembg_session_local):
    logger.info(f"Preprocessing image with bg_choice: {background_choice}, ratio: {foreground_ratio}")
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if background_choice == "Alpha as mask":
        logger.info("Using alpha channel as mask.")
        background_transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background_transparent, image)
    else:
        logger.info("Attempting automatic background removal.")
        image = remove_background(image, rembg_session_local=rembg_session_local, force=True)

    if foreground_ratio != 1.0:
        logger.info(f"Resizing content with scale_rate: {foreground_ratio}")
        image = do_resize_content(image, foreground_ratio)

    logger.info("Expanding image to square.")
    image = expand_to_square(image, bg_color=(0, 0, 0, 0))

    logger.info(f"Adding solid background color: {background_color}")
    image = add_background(image, background_color)

    logger.info("Converting final image to RGB.")
    return image.convert("RGB")

# --- NEW/Modified Helper Functions ---
async def send_status_message(websocket: WebSocket, status: str, error: str = None):
    """Helper to send JSON status updates ('processing', 'completed', 'failed')."""
    message = {"type": "status", "status": status}
    if error:
        message["error"] = error
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.warning("Client disconnected before status could be sent.")
    except Exception as e:
        logger.error(f"Error sending status message: {e}")

async def send_result_part_info(websocket: WebSocket, name: str, content_type: str):
    """Helper to send JSON identifying the next binary part."""
    message = {"type": "result_part", "name": name, "content_type": content_type}
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.warning(f"Client disconnected before sending info for {name}.")
        raise # Re-raise to stop further processing if info fails
    except Exception as e:
        logger.error(f"Error sending result part info for {name}: {e}")
        raise # Re-raise as this is critical for client interpretation

# --- Model Loading Function (Keep as is) ---
def load_models():
    """Loads all AI models and necessary components."""
    global models_loaded, crm_model, pipeline, rembg_session
    try:
        logger.info("Loading Rembg session...")
        rembg_session = rembg.new_session()
        logger.info("Rembg session loaded.")

        logger.info(f"Loading CRM model from {CRM_MODEL_PATH}...")
        if not os.path.exists(CRM_MODEL_PATH):
            raise FileNotFoundError(f"CRM model not found at {CRM_MODEL_PATH}")
        if not os.path.exists(SPECS_PATH):
            raise FileNotFoundError(f"Specs file not found at {SPECS_PATH}")
        specs = json.load(open(SPECS_PATH))
        crm_model = CRM(specs).to("cuda")
        crm_model.load_state_dict(torch.load(CRM_MODEL_PATH, map_location="cuda"), strict=False)
        crm_model.eval()
        logger.info("CRM model loaded successfully.")

        logger.info("Loading Pipeline models...")
        if not os.path.exists(STAGE1_CONFIG_PATH) or not os.path.exists(STAGE2_CONFIG_PATH):
            raise FileNotFoundError("Pipeline config files not found.")
        if not os.path.exists(PIXEL_DIFFUSION_PATH) or not os.path.exists(XYZ_DIFFUSION_PATH):
            raise FileNotFoundError("Pipeline diffusion model files not found.")

        stage1_config = OmegaConf.load(STAGE1_CONFIG_PATH).config
        stage2_config = OmegaConf.load(STAGE2_CONFIG_PATH).config
        stage1_config.models.resume = PIXEL_DIFFUSION_PATH
        stage2_config.models.resume = XYZ_DIFFUSION_PATH

        pipeline = TwoStagePipeline(
            stage1_model_config=stage1_config.models,
            stage2_model_config=stage2_config.models,
            stage1_sampler_config=stage1_config.sampler,
            stage2_sampler_config=stage2_config.sampler,
        )
        logger.info("Pipeline loaded successfully.")

        models_loaded = True
        logger.info("All models loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        models_loaded = False
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        models_loaded = False

# --- FastAPI Startup Event (Keep as is) ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading models...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_models)
    if not models_loaded:
        logger.error("MODEL LOADING FAILED. The service will not be able to process requests.")
    else:
        logger.info("Models loaded and ready.")


# --- WebSocket Endpoint (MODIFIED) ---
@app.websocket("/generate3d")
async def websocket_generate3d(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    if not models_loaded:
        logger.error("Models not loaded, refusing connection.")
        await send_status_message(websocket, "failed", "Server error: Models are not loaded.")
        await websocket.close(code=1011) # Internal Server Error
        return

    input_image_bytes = None
    obj_zip_path = None # Keep track of the generated zip path for potential cleanup

    try:
        # 1. Receive Image Data
        logger.info("Waiting for image data...")
        input_image_bytes = await websocket.receive_bytes()
        logger.info(f"Received image data: {len(input_image_bytes)} bytes.")

        # 2. Send Processing Status
        await send_status_message(websocket, "processing")

        # 3. Execute AI processing (Protected by Lock)
        async with gpu_lock:
            logger.info("Acquired GPU lock. Starting AI processing.")
            processed_ok = False # Flag to track success within the lock
            try:
                # Load image from bytes
                try:
                    input_image = Image.open(io.BytesIO(input_image_bytes))
                    input_image = input_image.convert("RGBA")
                except Exception as e:
                    logger.error(f"Failed to decode image: {e}")
                    raise ValueError(f"Invalid image data: {e}") from e

                # Preprocess image
                logger.info("Preprocessing image...")
                preprocessed_image = preprocess_image(
                    input_image,
                    DEFAULT_BG_CHOICE,
                    DEFAULT_FOREGROUND_RATIO,
                    DEFAULT_BACKGROUND_COLOR,
                    rembg_session
                ) # Returns RGB

                # Run pipeline
                logger.info("Running Stage 1 & 2 pipeline...")
                with torch.no_grad():
                    rt_dict = pipeline(preprocessed_image, scale=DEFAULT_SCALE, step=DEFAULT_STEP)
                np_imgs_combined = np.concatenate(rt_dict["stage1_images"], 1)
                np_xyzs_combined = np.concatenate(rt_dict["stage2_images"], 1)
                logger.info("Pipeline completed.")

                # --- Send Pixel Images PNG ---
                logger.info(f"Preparing {OUTPUT_PIXEL_PNG}...")
                pixel_png_buffer = io.BytesIO()
                Image.fromarray(np_imgs_combined).save(pixel_png_buffer, format="PNG")
                pixel_png_buffer.seek(0)
                await send_result_part_info(websocket, OUTPUT_PIXEL_PNG, "image/png")
                await websocket.send_bytes(pixel_png_buffer.getvalue())
                logger.info(f"Sent {OUTPUT_PIXEL_PNG}.")
                pixel_png_buffer.close() # Release memory

                # --- Send XYZ Images PNG ---
                logger.info(f"Preparing {OUTPUT_XYZ_PNG}...")
                xyz_png_buffer = io.BytesIO()
                Image.fromarray(np_xyzs_combined).save(xyz_png_buffer, format="PNG")
                xyz_png_buffer.seek(0)
                await send_result_part_info(websocket, OUTPUT_XYZ_PNG, "image/png")
                await websocket.send_bytes(xyz_png_buffer.getvalue())
                logger.info(f"Sent {OUTPUT_XYZ_PNG}.")
                xyz_png_buffer.close() # Release memory

                # --- Generate 3D Output ---
                logger.info(f"Generating 3D output...")
                # Assume generate3d returns (_, path_to_zip) and might create temporary files
                # We don't use tempfile.TemporaryDirectory here unless we modify generate3d
                # to explicitly write into it. We rely on generate3d's output path.
                with torch.no_grad():
                    # Make sure this returns the path to the final ZIP file
                    _, obj_zip_path = generate3d(crm_model, np_imgs_combined, np_xyzs_combined, "cuda")

                if not obj_zip_path or not os.path.exists(obj_zip_path):
                    raise RuntimeError("generate3d did not produce the expected output zip file or return its path.")
                logger.info(f"3D generation completed. Output zip path: {obj_zip_path}")

                # --- Send 3D Output ZIP ---
                logger.info(f"Preparing {OUTPUT_3D_ZIP}...")
                await send_result_part_info(websocket, OUTPUT_3D_ZIP, "application/zip")
                # Read the generated zip file and send its bytes
                try:
                    with open(obj_zip_path, "rb") as f:
                        zip_content = f.read()
                    await websocket.send_bytes(zip_content)
                    logger.info(f"Sent {OUTPUT_3D_ZIP} ({len(zip_content)} bytes).")
                except FileNotFoundError:
                    logger.error(f"Generated zip file not found at path: {obj_zip_path}")
                    raise RuntimeError(f"Failed to read generated 3D zip.")
                except Exception as e:
                    logger.error(f"Error reading or sending zip file {obj_zip_path}: {e}")
                    raise

                processed_ok = True # Mark as successful within the lock

            except (ValueError, RuntimeError, WebSocketDisconnect) as e: # Catch expected errors during processing/sending
                logger.error(f"Error during AI processing or sending parts: {e}", exc_info=True if not isinstance(e, WebSocketDisconnect) else False)
                # If an error occurs *during* processing, send failed status immediately
                # Check if connection is still active before sending
                if websocket.client_state == websocket.client_state.CONNECTED:
                     await send_status_message(websocket, "failed", f"Processing error: {str(e)[:200]}") # Limit error message length
                # No need to release lock explicitly, 'async with' handles it
                # Exit the websocket handler after failure
                return
            except Exception as e: # Catch unexpected errors
                logger.error(f"Unexpected error during AI processing: {e}", exc_info=True)
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await send_status_message(websocket, "failed", "An internal server error occurred during processing.")
                return # Exit after failure
            finally:
                logger.info("AI processing section finished. Releasing GPU lock.")
                # --- Cleanup the generated ZIP file ---
                # IMPORTANT: This assumes obj_zip_path points to a temporary file that
                # generate3d created and won't clean up itself. Adjust if generate3d
                # has different behavior (e.g., uses its own temp dir).
                if obj_zip_path and os.path.exists(obj_zip_path):
                    try:
                        # If generate3d consistently uses a temp dir, this might be safer:
                        # if tempfile.gettempdir() in os.path.abspath(obj_zip_path):
                        os.remove(obj_zip_path)
                        logger.info(f"Cleaned up temporary file: {obj_zip_path}")
                        # else:
                        #    logger.warning(f"Skipping cleanup of non-temporary file: {obj_zip_path}")
                    except OSError as e:
                        logger.error(f"Error cleaning up temporary file {obj_zip_path}: {e}")
                # Lock is released automatically by 'async with'

        # 4. Send Completed Status (Only if all parts were sent successfully)
        if processed_ok:
            await send_status_message(websocket, "completed")
            logger.info("Completion status sent.")

    except WebSocketDisconnect:
        logger.info("Client disconnected during operation.")
    except ValueError as e: # Catch specific errors like image decoding before lock
        logger.error(f"Value error before processing: {e}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await send_status_message(websocket, "failed", str(e))
    except Exception as e: # Catch errors outside the lock or re-raised errors
        logger.error(f"An unexpected error occurred outside processing lock: {e}", exc_info=True)
        if websocket.client_state == websocket.client_state.CONNECTED:
            await send_status_message(websocket, "failed", "An internal server error occurred.")
    finally:
        # 5. Close Connection
        logger.info("Closing WebSocket connection.")
        try:
            # Check state before closing, as errors might have already closed it
            if websocket.client_state == websocket.client_state.CONNECTED:
                 await websocket.close()
        except RuntimeError as e:
            if "WebSocket is not connected" in str(e):
                logger.info("WebSocket connection was already closed.")
            else:
                 logger.error(f"Error closing websocket: {e}", exc_info=True) # Log other runtime errors during close
        # Clean up large objects
        input_image_bytes = None


# --- Main Execution (Keep as is) ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    # Recommended: uvicorn main:app --host 0.0.0.0 --port 8001 --ws-ping-interval 20 --ws-ping-timeout 20
    uvicorn.run(app, host="0.0.0.0", port=8001)