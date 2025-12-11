#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a placeholder icon for HyperCoast QGIS Plugin.

Run this script to generate the plugin icon.
Requires: Pillow (pip install Pillow)
"""

try:
    from PIL import Image, ImageDraw, ImageFont

    # Create a 64x64 icon
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a gradient background (ocean-like colors)
    for y in range(size):
        ratio = y / size
        r = int(0 + (30 - 0) * ratio)
        g = int(100 + (60 - 100) * ratio)
        b = int(200 + (140 - 200) * ratio)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    # Draw spectral wave lines
    import math

    for offset, color in [
        (0, (255, 100, 100, 200)),
        (5, (100, 255, 100, 200)),
        (10, (100, 100, 255, 200)),
    ]:
        points = []
        for x in range(size):
            y = 32 + int(10 * math.sin((x + offset * 5) / 8)) + offset - 5
            points.append((x, y))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)

    # Draw border
    draw.rectangle(
        [(0, 0), (size - 1, size - 1)], outline=(255, 255, 255, 200), width=2
    )

    # Save
    img.save("hypercoast.png")
    print("Icon created: hypercoast.png")

except ImportError:
    print("Pillow not installed. Creating a placeholder file.")
    # Create a minimal 1x1 PNG as placeholder
    import struct
    import zlib

    def create_minimal_png():
        """Create a minimal valid PNG file."""
        # PNG signature
        signature = b"\x89PNG\r\n\x1a\n"

        # IHDR chunk
        width = 64
        height = 64
        bit_depth = 8
        color_type = 6  # RGBA
        ihdr_data = struct.pack(
            ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0
        )
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr_chunk = (
            struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        )

        # IDAT chunk (blue-green gradient, simplified)
        raw_data = b""
        for y in range(height):
            raw_data += b"\x00"  # filter type
            for x in range(width):
                r = 30
                g = int(100 - 40 * y / height)
                b = int(200 - 60 * y / height)
                raw_data += bytes([r, g, b, 255])

        compressed = zlib.compress(raw_data, 9)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat_chunk = (
            struct.pack(">I", len(compressed))
            + b"IDAT"
            + compressed
            + struct.pack(">I", idat_crc)
        )

        # IEND chunk
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

        return signature + ihdr_chunk + idat_chunk + iend_chunk

    with open("hypercoast.png", "wb") as f:
        f.write(create_minimal_png())
    print("Placeholder icon created: hypercoast.png")
