#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create icons for HyperCoast QGIS Plugin.

Run this script to generate all plugin icons.
Requires: Pillow (pip install Pillow)
"""

import math
import os

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not installed. Please run: pip install Pillow")
    exit(1)


def create_gradient_background(draw, size, colors):
    """Create a gradient background."""
    start_color, end_color = colors
    for y in range(size):
        ratio = y / size
        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))


def create_hypercoast_icon(size=64):
    """Create the main HyperCoast icon with spectral waves."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Ocean-like gradient background
    create_gradient_background(draw, size, ((0, 100, 200), (30, 60, 140)))

    # Draw spectral wave lines
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

    return img


def create_load_data_icon(size=64):
    """Create icon for Load Hyperspectral Data - folder with spectrum."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Deep teal gradient background
    create_gradient_background(draw, size, ((20, 80, 100), (10, 50, 70)))

    # Draw folder shape
    folder_color = (255, 200, 80, 255)
    folder_outline = (200, 150, 50, 255)

    # Folder back
    draw.rectangle(
        [(8, 18), (56, 52)], fill=folder_color, outline=folder_outline, width=2
    )

    # Folder tab
    draw.polygon(
        [(8, 18), (8, 12), (28, 12), (32, 18)],
        fill=folder_color,
        outline=folder_outline,
    )

    # Draw spectral bars inside folder (representing hyperspectral data)
    bar_colors = [
        (180, 50, 50, 255),  # Red
        (200, 100, 50, 255),  # Orange
        (200, 180, 50, 255),  # Yellow
        (50, 180, 50, 255),  # Green
        (50, 150, 200, 255),  # Cyan
        (80, 80, 200, 255),  # Blue
        (150, 50, 200, 255),  # Violet
    ]

    bar_width = 5
    bar_start_x = 14
    for i, color in enumerate(bar_colors):
        x = bar_start_x + i * (bar_width + 1)
        height = 10 + int(8 * math.sin(i * 0.8 + 1))
        draw.rectangle([(x, 48 - height), (x + bar_width - 1, 48)], fill=color)

    # Draw border
    draw.rectangle(
        [(0, 0), (size - 1, size - 1)], outline=(255, 255, 255, 200), width=2
    )

    return img


def create_band_combination_icon(size=64):
    """Create icon for Band Combination - RGB layers stacked."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark gray gradient background
    create_gradient_background(draw, size, ((60, 60, 70), (30, 30, 40)))

    # Draw three overlapping color rectangles (RGB bands)
    # Red band (back)
    draw.rectangle(
        [(12, 8), (44, 40)],
        fill=(220, 60, 60, 200),
        outline=(255, 100, 100, 255),
        width=2,
    )

    # Green band (middle)
    draw.rectangle(
        [(18, 14), (50, 46)],
        fill=(60, 180, 60, 200),
        outline=(100, 220, 100, 255),
        width=2,
    )

    # Blue band (front)
    draw.rectangle(
        [(24, 20), (56, 52)],
        fill=(60, 100, 220, 200),
        outline=(100, 140, 255, 255),
        width=2,
    )

    # Draw "RGB" text indicator with small circles
    draw.ellipse([(10, 50), (18, 58)], fill=(255, 80, 80, 255))
    draw.ellipse([(22, 50), (30, 58)], fill=(80, 220, 80, 255))
    draw.ellipse([(34, 50), (42, 58)], fill=(80, 120, 255, 255))

    # Draw border
    draw.rectangle(
        [(0, 0), (size - 1, size - 1)], outline=(255, 255, 255, 200), width=2
    )

    return img


def create_spectral_inspector_icon(size=64):
    """Create icon for Spectral Inspector - crosshair with spectral curve."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark purple gradient background
    create_gradient_background(draw, size, ((50, 30, 80), (30, 20, 50)))

    # Draw spectral curve (like a spectrogram)
    curve_color = (100, 255, 180, 255)
    points = []
    for x in range(8, 56):
        # Create a spectral response curve shape
        t = (x - 8) / 48
        y = 48 - int(
            25 * math.exp(-((t - 0.5) ** 2) * 8) * (1 + 0.3 * math.sin(t * 20))
        )
        points.append((x, y))

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=curve_color, width=2)

    # Draw crosshair/target indicator
    crosshair_color = (255, 220, 100, 255)
    cx, cy = 32, 32

    # Crosshair lines
    draw.line([(cx - 12, cy), (cx - 4, cy)], fill=crosshair_color, width=2)
    draw.line([(cx + 4, cy), (cx + 12, cy)], fill=crosshair_color, width=2)
    draw.line([(cx, cy - 12), (cx, cy - 4)], fill=crosshair_color, width=2)
    draw.line([(cx, cy + 4), (cx, cy + 12)], fill=crosshair_color, width=2)

    # Center circle
    draw.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], outline=crosshair_color, width=2)

    # Draw axes
    axis_color = (150, 150, 150, 200)
    draw.line([(8, 52), (56, 52)], fill=axis_color, width=1)  # X axis
    draw.line([(8, 52), (8, 12)], fill=axis_color, width=1)  # Y axis

    # Draw border
    draw.rectangle(
        [(0, 0), (size - 1, size - 1)], outline=(255, 255, 255, 200), width=2
    )

    return img


def create_about_icon(size=64):
    """Create icon for About - uses the main hypercoast icon with info symbol."""
    # Just use the main hypercoast icon for About
    return create_hypercoast_icon(size)


def main():
    """Generate all icons."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    icons = {
        "hypercoast.png": create_hypercoast_icon,
        "load_data.png": create_load_data_icon,
        "band_combination.png": create_band_combination_icon,
        "spectral_inspector.png": create_spectral_inspector_icon,
    }

    for filename, creator_func in icons.items():
        filepath = os.path.join(script_dir, filename)
        img = creator_func()
        img.save(filepath)
        print(f"Icon created: {filename}")

    print(f"\nAll icons saved to: {script_dir}")


if __name__ == "__main__":
    main()
