#!/bin/bash
# --- Configuration ---
PROJECT_DIR="/home/fyz/CRM"       # Your project directory containing http_server.py
CONDA_ENV_NAME="crm"              # The name of your Conda environment
PYTHON_SCRIPT_NAME="http_server"  # The name of your Python file (without .py)
FASTAPI_APP_VARIABLE="app"        # The variable holding the FastAPI() instance in your script
HOST="0.0.0.0"                    # Host to bind the server to
PORT="8001"                       # Port to run the server on
USE_RELOAD="true"                 # Set to "true" for development (--reload), "false" for production

# --- Script Logic ---

echo "--------------------------------------------------"
echo "Attempting to start CRM FastAPI Server..."
echo "--------------------------------------------------"
echo "- Project Directory: $PROJECT_DIR"
echo "- Conda Environment: $CONDA_ENV_NAME"
echo "- Python Script: ${PYTHON_SCRIPT_NAME}.py"
echo "- Host: $HOST"
echo "- Port: $PORT"
echo "- Development Reload: $USE_RELOAD"
echo "--------------------------------------------------"

# Navigate to the project directory
echo "1. Changing directory to $PROJECT_DIR..."
cd "$PROJECT_DIR" || { echo "Error: Failed to change directory to $PROJECT_DIR. Exiting."; exit 1; }
echo "   Current directory: $(pwd)"

# Construct the uvicorn command
UVICORN_CMD="uvicorn ${PYTHON_SCRIPT_NAME}:${FASTAPI_APP_VARIABLE} --host $HOST --port $PORT"
if [ "$USE_RELOAD" = "true" ]; then
  UVICORN_CMD="$UVICORN_CMD --reload"
  echo "2. Starting Uvicorn with auto-reload enabled..."
else
  echo "2. Starting Uvicorn for production (no auto-reload)..."
fi

# Execute uvicorn within the specified Conda environment
# Using 'conda run' is generally the recommended way for scripts
echo "   Executing: conda run -n $CONDA_ENV_NAME $UVICORN_CMD"
echo "--------------------------------------------------"

conda run -n "$CONDA_ENV_NAME" $UVICORN_CMD

# Note: The script will stay here until uvicorn is stopped (e.g., with CTRL+C)

EXIT_CODE=$?
echo "--------------------------------------------------"
echo "Uvicorn server process finished with exit code: $EXIT_CODE"
echo "--------------------------------------------------"

exit $EXIT_CODE