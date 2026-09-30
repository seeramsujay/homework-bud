"""
Photorealistic Phone Scan & Royal Blue Ink Renderer for Homework-Bud
Overlays synthesized handwriting onto blank ruled notebook photos with physical ink blending.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from ruled_sheet_detector import PageLayout, RuledSheetDetector
import drawing


class ScanRenderer:
    def __init__(
        self,
        ink_rgb: Tuple[int, int, int] = (48, 87, 163),   # Royal Blue extracted from pen sample (#3057a3)
        ink_bleed: float = 0.3,                          # Microscopic bleed
        human_jitter: float = 0.6,                       # Subtle human baseline jitter
        high_contrast: bool = True                       # High contrast scanner style
    ):
        self.ink_rgb = ink_rgb
        self.ink_bleed = ink_bleed
        self.human_jitter = human_jitter
        self.high_contrast = high_contrast

    @classmethod
    def sample_ink_from_image(cls, crop_image_path: str) -> Tuple[int, int, int]:
        """
        Samples the median royal blue ink color from an uploaded handwritten sample photo.
        """
        img = cv2.imread(crop_image_path)
        if img is None:
            return (48, 87, 163)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] <= 180)
        blue_pixels = img[blue_mask]
        if len(blue_pixels) == 0:
            return (48, 87, 163)
        median_bgr = np.median(blue_pixels, axis=0).astype(int)
        return (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))

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

        # Target height relative to line spacing
        target_char_height = layout.line_spacing * 0.58
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

            raw_xs = coords[:, 0]
            raw_ys = -1.0 * coords[:, 1]
            raw_eoses = coords[:, 2]

            min_x, max_x = np.min(raw_xs), np.max(raw_xs)
            min_y, max_y = np.min(raw_ys), np.max(raw_ys)
            orig_height = max(1.0, max_y - min_y)
            orig_width = max(1.0, max_x - min_x)

            scale = target_char_height / orig_height
            baseline_y = layout.baselines[line_idx]

            jitter_y = np.random.uniform(-self.human_jitter, self.human_jitter)
            jitter_x = np.random.uniform(-self.human_jitter * 2, self.human_jitter * 2)

            scaled_xs = (raw_xs - min_x) * scale + layout.left_margin + jitter_x
            scaled_ys = (raw_ys - max_y) * scale + baseline_y + jitter_y

            if np.max(scaled_xs) > layout.right_margin:
                overflow_scale = (layout.right_margin - layout.left_margin) / (orig_width * scale)
                if overflow_scale < 1.0:
                    scale *= overflow_scale
                    scaled_xs = (raw_xs - min_x) * scale + layout.left_margin + jitter_x
                    scaled_ys = (raw_ys - max_y) * scale + baseline_y + jitter_y

            self._draw_solid_ink_strokes(ink_layer, scaled_xs, scaled_ys, raw_eoses, scale)

        # Multiply-blend solid ink layer onto the paper photo
        blended_bgr = self._multiply_blend(bg_bgr, ink_layer)

        # Apply high-contrast scan post-processing if enabled
        if self.high_contrast:
            blended_bgr = self._apply_scan_contrast(blended_bgr)

        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
            cv2.imwrite(output_image_path, blended_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])

        return blended_bgr

    def _draw_solid_ink_strokes(
        self,
        ink_layer: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        eoses: np.ndarray,
        scale: float
    ):
        """
        Draws uniform, solid royal blue ink strokes without artificial fading.
        """
        base_r, base_g, base_b = self.ink_rgb
        h, w = ink_layer.shape[:2]
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        stroke_start = 0

        # Consistent pen stroke width
        thickness = max(2, int(round(2.0 * (scale / 1.0))))

        for i in range(len(eoses)):
            if eoses[i] == 1.0 or i == len(eoses) - 1:
                stroke_segment = pts[stroke_start:i + 1]
                stroke_start = i + 1

                if len(stroke_segment) < 2:
                    continue

                for p in range(len(stroke_segment) - 1):
                    p1 = (int(round(stroke_segment[p][0])), int(round(stroke_segment[p][1])))
                    p2 = (int(round(stroke_segment[p + 1][0])), int(round(stroke_segment[p + 1][1])))

                    if p1[0] < 0 or p1[0] >= w or p1[1] < 0 or p1[1] >= h:
                        continue

                    # Solid uniform ink opacity (255)
                    cv2.line(ink_layer, p1, p2, (base_b, base_g, base_r, 255), thickness, cv2.LINE_AA)

    def _multiply_blend(self, bg_bgr: np.ndarray, ink_rgba: np.ndarray) -> np.ndarray:
        """
        Physically blends ink into paper using Multiply blending.
        """
        ink_bgr = ink_rgba[:, :, :3].astype(np.float32)
        ink_alpha = (ink_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        bg_float = bg_bgr.astype(np.float32)
        multiplied = (bg_float * ink_bgr) / 255.0
        result = bg_float * (1.0 - ink_alpha) + multiplied * ink_alpha

        if self.ink_bleed > 0:
            result = cv2.GaussianBlur(result, (3, 3), self.ink_bleed)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_scan_contrast(self, img_bgr: np.ndarray, gamma: float = 1.35, contrast_alpha: float = 1.25) -> np.ndarray:
        """
        Elevates contrast and gamma to mimic phone document scanning apps (CamScanner/Adobe Scan).
        """
        # Gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(img_bgr, table)

        # High contrast linear scaling
        contrast_img = cv2.convertScaleAbs(gamma_img, alpha=contrast_alpha, beta=-15)
        return contrast_img

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
