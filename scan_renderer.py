"""
Photorealistic Phone Scan & Royal Blue Ink Renderer for Homework-Bud
Overlays synthesized handwriting onto blank ruled notebook photos with physical ink blending.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple, Optional
from ruled_sheet_detector import PageLayout, RuledSheetDetector
import drawing


class ScanRenderer:
    def __init__(
        self,
        ink_rgb: Tuple[int, int, int] = (26, 62, 175),  # Classic Royal Blue
        ink_bleed: float = 0.5,                          # Microscopic bleed into paper fibers
        human_jitter: float = 0.8,                        # Slight human variation per line
    ):
        self.ink_rgb = ink_rgb
        self.ink_bleed = ink_bleed
        self.human_jitter = human_jitter

    @classmethod
    def sample_ink_from_image(cls, crop_image_path: str) -> Tuple[int, int, int]:
        """
        Samples the median darkest ink color from an uploaded ink swatch/crop photo.
        """
        img = cv2.imread(crop_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load ink image at {crop_image_path}")
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find pixels that are colored (not white paper background)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink_pixels = img_rgb[gray < 200]
        if len(ink_pixels) == 0:
            return (26, 62, 175)
        # Take median color of ink pixels
        median_color = np.median(ink_pixels, axis=0).astype(int)
        return tuple(median_color.tolist())

    def render_page(
        self,
        blank_sheet_path: str,
        lines_stroke_coords: List[np.ndarray],
        layout: Optional[PageLayout] = None,
        output_image_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders a page of handwritten strokes directly onto the blank paper photo.
        """
        bg_bgr = cv2.imread(blank_sheet_path)
        if bg_bgr is None:
            raise FileNotFoundError(f"Could not load paper image at {blank_sheet_path}")

        h, w = bg_bgr.shape[:2]

        if layout is None:
            detector = RuledSheetDetector()
            layout = detector.analyze_image(blank_sheet_path)

        # Create transparent RGBA ink layer
        ink_layer = np.zeros((h, w, 4), dtype=np.uint8)

        # Target height for x-height / capitals relative to line spacing
        target_char_height = layout.line_spacing * 0.55
        available_lines = len(layout.baselines)

        for line_idx, stroke_offsets in enumerate(lines_stroke_coords):
            if line_idx >= available_lines:
                break  # Page is full, remainder goes to next page

            if len(stroke_offsets) == 0:
                continue

            # Convert offsets to coordinates
            coords = drawing.offsets_to_coords(stroke_offsets)
            coords = drawing.denoise(coords)
            coords[:, :2] = drawing.align(coords[:, :2])

            # Invert Y so up is positive in standard screen coordinates
            raw_xs = coords[:, 0]
            raw_ys = -1.0 * coords[:, 1]
            raw_eoses = coords[:, 2]

            min_x, max_x = np.min(raw_xs), np.max(raw_xs)
            min_y, max_y = np.min(raw_ys), np.max(raw_ys)
            orig_height = max(1.0, max_y - min_y)
            orig_width = max(1.0, max_x - min_x)

            # Scale factor to sit naturally on ruled line
            scale = target_char_height / orig_height

            # Natural human baseline: align near the bottom baseline
            baseline_y = layout.baselines[line_idx]

            # Add subtle human jitter
            jitter_y = np.random.uniform(-self.human_jitter, self.human_jitter)
            jitter_x = np.random.uniform(-self.human_jitter * 2, self.human_jitter * 2)

            scaled_xs = (raw_xs - min_x) * scale + layout.left_margin + jitter_x
            # Position so baseline sits directly on the ruled line
            # In English writing, baseline is ~75% down from top of bounding box (leaving room for descenders)
            scaled_ys = (raw_ys - max_y) * scale + baseline_y + jitter_y

            # Check if line fits within right margin
            if np.max(scaled_xs) > layout.right_margin:
                overflow_scale = (layout.right_margin - layout.left_margin) / (orig_width * scale)
                if overflow_scale < 1.0:
                    scale *= overflow_scale
                    scaled_xs = (raw_xs - min_x) * scale + layout.left_margin + jitter_x
                    scaled_ys = (raw_ys - max_y) * scale + baseline_y + jitter_y

            # Draw strokes with royal blue pen physics
            self._draw_pen_strokes(ink_layer, scaled_xs, scaled_ys, raw_eoses, scale)

        # Multiply-blend ink layer onto the paper photo
        blended_bgr = self._multiply_blend(bg_bgr, ink_layer)

        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
            cv2.imwrite(output_image_path, blended_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return blended_bgr

    def _draw_pen_strokes(
        self,
        ink_layer: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        eoses: np.ndarray,
        scale: float
    ):
        """
        Draws realistic royal blue ballpoint/gel pen strokes with pressure & shading.
        """
        base_r, base_g, base_b = self.ink_rgb
        h, w = ink_layer.shape[:2]

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        stroke_start = 0

        # Base pen stroke width in pixels (typical 0.5mm pen on phone camera resolution)
        nominal_width = max(1.8, 1.8 * (scale / 1.0))

        for i in range(len(eoses)):
            if eoses[i] == 1.0 or i == len(eoses) - 1:
                # Segment of one continuous stroke
                stroke_segment = pts[stroke_start:i + 1]
                stroke_start = i + 1

                if len(stroke_segment) < 2:
                    continue

                for p in range(len(stroke_segment) - 1):
                    p1 = (int(stroke_segment[p][0]), int(stroke_segment[p][1]))
                    p2 = (int(stroke_segment[p + 1][0]), int(stroke_segment[p + 1][1]))

                    if p1[0] < 0 or p1[0] >= w or p1[1] < 0 or p1[1] >= h:
                        continue

                    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    # Speed variation: faster movements deposit slightly less ink
                    speed_factor = np.clip(1.0 - (dist / 40.0) * 0.15, 0.75, 1.0)
                    thickness = max(1, int(round(nominal_width * speed_factor)))

                    # Color with slight dynamic variation
                    shade_var = np.random.uniform(0.92, 1.05)
                    r = int(np.clip(base_r * shade_var, 0, 255))
                    g = int(np.clip(base_g * shade_var, 0, 255))
                    b = int(np.clip(base_b * shade_var, 0, 255))
                    alpha = int(240 * speed_factor)

                    # BGR order for OpenCV
                    cv2.line(ink_layer, p1, p2, (b, g, r, alpha), thickness, cv2.LINE_AA)

    def _multiply_blend(self, bg_bgr: np.ndarray, ink_rgba: np.ndarray) -> np.ndarray:
        """
        Physically blends ink into paper using Multiply blending.
        """
        ink_bgr = ink_rgba[:, :, :3].astype(np.float32)
        ink_alpha = (ink_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        bg_float = bg_bgr.astype(np.float32)

        # Multiply blend formula: (Background * Ink) / 255
        multiplied = (bg_float * ink_bgr) / 255.0

        # Interpolate between original paper and multiplied ink based on ink alpha
        result = bg_float * (1.0 - ink_alpha) + multiplied * ink_alpha

        # Micro-softening to simulate ink fiber absorption
        if self.ink_bleed > 0:
            result = cv2.GaussianBlur(result, (3, 3), self.ink_bleed)

        return np.clip(result, 0, 255).astype(np.uint8)

    @classmethod
    def compile_pdf(cls, image_paths: List[str], output_pdf_path: str):
        """
        Compiles multiple rendered page images into a single multi-page PDF document.
        """
        if not image_paths:
            raise ValueError("No images provided for PDF compilation")

        pil_images = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            pil_images.append(img)

        os.makedirs(os.path.dirname(output_pdf_path) or '.', exist_ok=True)
        pil_images[0].save(
            output_pdf_path,
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else [],
            quality=95
        )
        print(f"[OK] Successfully compiled {len(image_paths)} pages to {output_pdf_path}")
