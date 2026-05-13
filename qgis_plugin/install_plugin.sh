#!/bin/bash
# Script to package the HyperCoast QGIS Plugin for distribution.
#
# Produces ``dist/hypercoast_<version>.zip`` with ``hypercoast`` as the
# top-level archive folder, matching the official QGIS plugin name. The
# source folder in this repository is ``hypercoast_qgis``.

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Plugin names: source folder vs. installed/published name
SOURCE_PLUGIN_NAME="hypercoast_qgis"
QGIS_PLUGIN_NAME="hypercoast"

# Read version from metadata.txt so it stays in sync with the plugin
METADATA_FILE="$SOURCE_PLUGIN_NAME/metadata.txt"
if [[ ! -f "$METADATA_FILE" ]]; then
    echo "Error: metadata file not found: $METADATA_FILE" >&2
    exit 1
fi
VERSION="$(awk -F '=' '/^[[:space:]]*version[[:space:]]*=/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$METADATA_FILE")"
if [[ -z "$VERSION" ]]; then
    echo "Error: could not determine version from $METADATA_FILE" >&2
    exit 1
fi

# Create output directory
OUTPUT_DIR="dist"
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for clean copy
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT
mkdir -p "$TEMP_DIR/$QGIS_PLUGIN_NAME"

# Copy plugin files from the source folder to the published folder name
cp -r "$SOURCE_PLUGIN_NAME"/* "$TEMP_DIR/$QGIS_PLUGIN_NAME/"

# Remove unnecessary files
find "$TEMP_DIR" -type f -name "*.pyc" -delete
find "$TEMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -type f -name "*.py.bak" -delete
find "$TEMP_DIR" -type f -name ".DS_Store" -delete

# Create ZIP archive
ZIP_NAME="${QGIS_PLUGIN_NAME}_${VERSION}.zip"
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$OUTPUT_DIR/$ZIP_NAME" "$QGIS_PLUGIN_NAME"

echo "Plugin packaged: $OUTPUT_DIR/$ZIP_NAME"
echo ""
echo "Installation instructions:"
echo "1. Open QGIS"
echo "2. Go to Plugins -> Manage and Install Plugins"
echo "3. Click 'Install from ZIP'"
echo "4. Select: $OUTPUT_DIR/$ZIP_NAME"
echo "5. Click 'Install Plugin'"
