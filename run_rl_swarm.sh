#!/bin/bash

set -euo pipefail

# General arguments
ROOT=$PWD

export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # 2 minutes
export SWARM_CONTRACT="0xFaD7C5e93f28257429569B854151A1B8DCD404c2"
export HUGGINGFACE_ACCESS_TOKEN="None"

DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}

# Docker permission workaround
if [ -n "$DOCKER" ]; then
    volumes=(
        /home/gensyn/rl_swarm/modal-login/temp-data
        /home/gensyn/rl_swarm/keys
        /home/gensyn/rl_swarm/configs
        /home/gensyn/rl_swarm/logs
    )
    for volume in ${volumes[@]}; do
        sudo chown -R 1001:1001 $volume
    done
fi

CPU_ONLY=${CPU_ONLY:-""}
ORG_ID=${ORG_ID:-""}

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RED_TEXT="\033[31m"
RESET_TEXT="\033[0m"

echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

echo_red() {
    echo -e "$RED_TEXT$1$RESET_TEXT"
}

# Function to clean up background processes upon exit
cleanup() {
    echo_green ">> Shutting down..."
    # Kill all processes belonging to this script's process group
    kill -- -$$ || true
    exit 0
}

errnotify() {
    echo_red ">> An error was detected while running rl-swarm. See $ROOT/logs for full logs."
}

trap cleanup EXIT
trap errnotify ERR

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██              ███████ ██      ██  █████  ██████  ███    ███
    ██   ██ ██              ██      ██      ██ ██   ██ ██   ██ ████  ████
    ██████  ██        █████ ███████ ██  █   ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                    ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████         ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn

EOF

mkdir -p "$ROOT/logs"

# ✅ KEY FIX 1: Login and Proxy Server Logic
# This block now robustly handles starting the proxy server and the login flow.
if [ "$CONNECT_TO_TESTNET" = true ]; then
    echo_green ">> Preparing Modal Proxy server..."
    cd modal-login

    # Install dependencies if they are missing
    if ! command -v node > /dev/null 2>&1; then
        echo "Node.js not found. Installing NVM and latest Node.js..."
        export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install node
    fi
    if ! command -v yarn > /dev/null 2>&1; then
        echo "Yarn not found. Installing..."
        npm install -g yarn
    fi
    if [ -z "$DOCKER" ]; then
        yarn install --immutable
        yarn build > "$ROOT/logs/yarn.log" 2>&1
    fi
    
    # Always start the server in the background. This fixes the "Connection refused" error.
    echo_green ">> Starting Modal Proxy server in the background..."
    yarn start >> "$ROOT/logs/yarn.log" 2>&1 &
    SERVER_PID=$!
    echo "Started server process: $SERVER_PID"
    cd ..
    sleep 5 # Give the server a moment to start up

    # Check for existing credentials or guide user through login
    if [ -f "modal-login/temp-data/userData.json" ]; then
        echo_green ">> Found existing login credentials."
        ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    else
        echo_green ">> Please log in via your browser at http://localhost:3000"
        # Wait for user to log in and for credentials file to be created
        while [ ! -f "modal-login/temp-data/userData.json" ]; do
            sleep 5
        done
        echo "Found userData.json. Proceeding..."
        ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
        
        # Wait for API key activation
        while true; do
            STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
            if [[ "$STATUS" == "activated" ]]; then
                echo "API key is activated! Proceeding..."
                break
            else
                sleep 5
            fi
        done
    fi
    echo "Your ORG_ID is set to: $ORG_ID"
fi

echo_green ">> Installing Python requirements..."
pip install --upgrade pip

# ✅ KEY FIX 2: Restored essential dependencies for training and inference
pip install git+https://github.com/xailong-6969/genrl.git
pip install reasoning-gym>=0.1.20
pip install trl
pip install vllm
pip install bitsandbytes 
pip install hivemind@git+https://github.com/gensyn-ai/hivemind@639c964a8019de63135a2594663b5bec8e5356dd

# Copy default config if needed
if [ ! -d "$ROOT/configs" ]; then
    mkdir "$ROOT/configs"
fi
if [ ! -f "$ROOT/configs/rg-swarm.yaml" ]; then
    cp "$ROOT/rgym_exp/config/rg-swarm.yaml" "$ROOT/configs/rg-swarm.yaml"
fi

echo_green ">> Setup complete!"

# Handle Hugging Face token
HF_TOKEN=${HF_TOKEN:-""}
if [ -z "${HF_TOKEN}" ]; then
    echo -en $GREEN_TEXT
    read -p ">> Would you like to push models to the Hugging Face Hub? [y/N] " yn
    echo -en $RESET_TEXT
    case ${yn:-N} in
        [Yy]*) read -p "Enter your Hugging Face access token: " HUGGINGFACE_ACCESS_TOKEN ;;
        *) HUGGINGFACE_ACCESS_TOKEN="None" ;;
    esac
else
    HUGGINGFACE_ACCESS_TOKEN=${HF_TOKEN}
fi

# ✅ KEY FIX 3: Removed the redundant interactive prompt for model name.
# Model selection is now handled exclusively by the YAML file and command-line overrides.
echo_green ">> Model will be selected based on your config file."
echo_green ">> You can override it from the command line, e.g., 'model_select=small_a'"
echo_green ">> Good luck in the swarm!"
echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"

python -m rgym_exp.runner.swarm_launcher \
    --config-path "$ROOT/rgym_exp/config" \
    --config-name "rg-swarm.yaml" 

wait # Keep script running until Ctrl+C
