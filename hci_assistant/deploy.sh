#!/bin/bash
# Deploy HCI Smart Home Assistant to SC171 device.
# Run from host: bash deploy.sh

set -e
DEVICE="fibo@10.0.0.102"
TARGET="/home/fibo/project/HCI/"
FILES="main.py config.py pipeline.py asr_engine.py wake_word.py llm_engine.py action_executor.py smart_home_state.py utils.py prompt_templates.py"

echo "=== Deploying to $DEVICE:$TARGET ==="

# Ensure WG tunnel is up
sudo wg-quick up /home/hiki/Code/school/aiemb/temp/temp6.conf 2>/dev/null || true

# Create target directories
ssh "$DEVICE" "mkdir -p $TARGET/logs $TARGET/output"

# Copy all source files
for f in $FILES; do
    echo "  $f"
    scp "$f" "${DEVICE}:${TARGET}"
done
scp deploy.sh "${DEVICE}:${TARGET}"

echo ""
echo "=== Deploy complete ==="
echo "Run on device:"
echo "  ssh $DEVICE"
echo "  cd $TARGET"
echo "  python3 main.py"
echo "  python3 main.py --debug   (verbose)"
