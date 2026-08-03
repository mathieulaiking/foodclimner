#!/bin/bash

# Archive either 'data' or 'out' directory to a tar.gz file
# Usage: archive_dir.sh <data|out>
# Output format: <data_or_out>-YYYYMMDD_<hostname>.tar.gz

# Check if argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <data|out>"
    exit 1
fi

DIR_TO_ARCHIVE="$1"

# Validate argument
if [ "$DIR_TO_ARCHIVE" != "data" ] && [ "$DIR_TO_ARCHIVE" != "out" ]; then
    echo "Error: Argument must be either 'data' or 'out'"
    exit 1
fi

# Check if directory exists from project root
if [ ! -d "$DIR_TO_ARCHIVE" ]; then
    echo "Error: Directory '$DIR_TO_ARCHIVE' not found in project root"
    exit 1
fi

# Get current date and time in YYYYMMDD and HHMMSS format
DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)

# Get machine hostname (convert to uppercase)
MACHINE=$(hostname | tr '[:lower:]' '[:upper:]')

# Create archive filename
ARCHIVE_NAME="${DIR_TO_ARCHIVE}-${DATE}-${TIME}_${MACHINE}.tar.gz"

# Create the archive
echo "Archiving '$DIR_TO_ARCHIVE' to '$ARCHIVE_NAME'..."
tar -czf "$ARCHIVE_NAME" "$DIR_TO_ARCHIVE"

if [ $? -eq 0 ]; then
    echo "✓ Archive created successfully: $ARCHIVE_NAME"
    ls -lh "$ARCHIVE_NAME"
else
    echo "Error: Failed to create archive"
    exit 1
fi

# Transfer archive based on machine location
LOCAL_PATH="/lustre/fsstor/projects/rech/xpc/urz45id/sf4cd"
REMOTE_PATH="jz:/lustre/fsstor/projects/rech/xpc/urz45id/sf4cd"

if [[ "$MACHINE" == *"JEAN-ZAY"* ]]; then
    # On Jean-Zay, use cp
    echo "On Jean-Zay cluster, copying archive locally..."
    cp "$ARCHIVE_NAME" "$LOCAL_PATH/"
    if [ $? -eq 0 ]; then
        echo "✓ Archive copied to $LOCAL_PATH/"
    else
        echo "Error: Failed to copy archive to $LOCAL_PATH"
        exit 1
    fi
else
    # Not on Jean-Zay, use scp
    echo "Transferring archive to Jean-Zay via scp..."
    scp "$ARCHIVE_NAME" "$REMOTE_PATH/"
    if [ $? -eq 0 ]; then
        echo "✓ Archive transferred to $REMOTE_PATH/"
    else
        echo "Error: Failed to transfer archive to $REMOTE_PATH"
        exit 1
    fi
fi
