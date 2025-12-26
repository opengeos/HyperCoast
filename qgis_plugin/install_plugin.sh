#!/bin/bash
# Script to package the HyperCoast QGIS Plugin for distribution

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Plugin name
PLUGIN_NAME="hypercoast_qgis"
VERSION="0.3.0"

# Create output directory
OUTPUT_DIR="dist"
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for clean copy
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/$PLUGIN_NAME"

# Copy plugin files
cp -r "$PLUGIN_NAME"/* "$TEMP_DIR/$PLUGIN_NAME/"

# Remove unnecessary files
find "$TEMP_DIR" -type f -name "*.pyc" -delete
find "$TEMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -type f -name "*.py.bak" -delete
find "$TEMP_DIR" -type f -name ".DS_Store" -delete

# Create ZIP archive
ZIP_NAME="${PLUGIN_NAME}_${VERSION}.zip"
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/$OUTPUT_DIR/$ZIP_NAME" "$PLUGIN_NAME"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Plugin packaged: $OUTPUT_DIR/$ZIP_NAME"
echo ""
echo "Installation instructions:"
echo "1. Open QGIS"
echo "2. Go to Plugins -> Manage and Install Plugins"
echo "3. Click 'Install from ZIP'"
echo "4. Select: $OUTPUT_DIR/$ZIP_NAME"
echo "5. Click 'Install Plugin'"

