"""
Homework-Bud End-to-End Homework Generation Engine
Generates multi-page handwritten homework on ruled paper photos in royal blue ink.
"""

import os
import glob
import textwrap
from typing import List, Optional, Tuple
import numpy as np

import drawing
from ruled_sheet_detector import RuledSheetDetector, PageLayout
from scan_renderer import ScanRenderer
from demo import Hand


class HomeworkEngine:
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        styles_dir: str = 'styles',
        ink_rgb: Tuple[int, int, int] = (26, 62, 175),  # Royal Blue
        bias: float = 0.85                               # High bias = neat, legible handwriting
    ):
        self.checkpoint_dir = checkpoint_dir
        self.styles_dir = styles_dir
        self.ink_rgb = ink_rgb
        self.bias = bias
        self.hand = None
        self.detector = RuledSheetDetector()
        self.renderer = ScanRenderer(ink_rgb=self.ink_rgb)

    def load_model(self):
        """Loads and initializes the Graves RNN model from checkpoints."""
        if self.hand is None:
            print(f"[INFO] Initializing Handwriting Synthesis RNN from {self.checkpoint_dir}...")
            self.hand = Hand(checkpoint_dir=self.checkpoint_dir)
            print("[OK] Model restored successfully from checkpoint.")

    def wrap_text_for_ruled_page(self, raw_text: str, max_chars_per_line: int = 60) -> List[str]:
        """
        Wraps long paragraphs into lines that comfortably fit within notebook margins.
        Preserves intentional line breaks (e.g. numbered questions, paragraphs).
        """
        valid_charset = set(drawing.alphabet)
        wrapped_lines = []

        paragraphs = raw_text.split('\n')
        for para in paragraphs:
            para = para.strip()
            if not para:
                wrapped_lines.append("")  # Empty line for paragraph gap
                continue

            # Clean characters not supported by alphabet
            cleaned_para = ""
            for ch in para:
                if ch in valid_charset:
                    cleaned_para += ch
                elif ch == '\t':
                    cleaned_para += "    "
                elif ch in ['“', '”']:
                    cleaned_para += '"'
                elif ch in ['‘', '’']:
                    cleaned_para += "'"
                elif ch in ['–', '—']:
                    cleaned_para += '-'
                elif ch in ['Q', 'X', 'Z']:
                    cleaned_para += ch.lower()
                else:
                    cleaned_para += ' '

            sub_lines = textwrap.wrap(cleaned_para, width=max_chars_per_line, break_long_words=False)
            if not sub_lines:
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(sub_lines)

        return wrapped_lines

    def generate_homework(
        self,
        text: str,
        blank_sheet_paths: List[str],
        output_dir: str = "output",
        style_id: Optional[int] = None,
        custom_style_prefix: Optional[str] = None,
        max_chars_per_line: int = 60
    ) -> Tuple[List[str], str]:
        """
        Generates photorealistic handwritten homework pages and compiles into PDF.
        """
        self.load_model()
        os.makedirs(output_dir, exist_ok=True)

        if not blank_sheet_paths:
            raise ValueError("At least one blank sheet image path must be provided.")

        # 1. Analyze first blank sheet to determine layout & available lines per page
        print(f"[INFO] Analyzing ruled sheet: {blank_sheet_paths[0]}...")
        layouts = []
        for p in blank_sheet_paths:
            layouts.append(self.detector.analyze_image(p))

        lines = self.wrap_text_for_ruled_page(text, max_chars_per_line=max_chars_per_line)
        print(f"[INFO] Total text lines to write: {len(lines)}")

        # 2. Synthesize stroke coordinates for all lines
        print(f"[INFO] Synthesizing handwriting strokes (bias={self.bias})...")
        biases = [self.bias] * len(lines)
        
        # Prepare style parameters
        styles = None
        if custom_style_prefix:
            # Custom style from user photo or touch recorder
            custom_strokes = np.load(f"{custom_style_prefix}-strokes.npy")
            chars_arr = np.load(f"{custom_style_prefix}-chars.npy")
            if chars_arr.dtype.kind in ('S', 'a', 'b'):
                custom_chars = chars_arr.tobytes().decode('utf-8', errors='ignore')
            elif hasattr(chars_arr, 'item'):
                custom_chars = str(chars_arr.item())
            else:
                custom_chars = str(chars_arr)
            custom_chars = custom_chars.strip('\x00').strip()
            # Sample with custom style
            stroke_lines = self._sample_with_custom_style(lines, biases, custom_strokes, custom_chars)
        elif style_id is not None:
            styles = [style_id] * len(lines)
            stroke_lines = self.hand._sample(lines, biases=biases, styles=styles)
        else:
            # Unconditioned style
            stroke_lines = self.hand._sample(lines, biases=biases, styles=None)

        # 3. Paginate lines onto blank sheets
        generated_page_images = []
        current_line_idx = 0
        page_num = 1

        while current_line_idx < len(lines):
            sheet_idx = (page_num - 1) % len(blank_sheet_paths)
            sheet_path = blank_sheet_paths[sheet_idx]
            layout = layouts[sheet_idx]

            available_lines_on_page = len(layout.baselines)
            page_stroke_lines = stroke_lines[current_line_idx : current_line_idx + available_lines_on_page]
            current_line_idx += available_lines_on_page

            page_img_path = os.path.join(output_dir, f"homework_page_{page_num:02d}.png")
            print(f"[INFO] Rendering Page {page_num} on {os.path.basename(sheet_path)}...")

            self.renderer.render_page(
                blank_sheet_path=sheet_path,
                lines_stroke_coords=page_stroke_lines,
                layout=layout,
                output_image_path=page_img_path
            )
            generated_page_images.append(page_img_path)
            page_num += 1

        # 4. Compile into multi-page PDF
        pdf_path = os.path.join(output_dir, "homework_assignment.pdf")
        self.renderer.compile_pdf(generated_page_images, pdf_path)

        print(f"[DONE] Successfully generated {len(generated_page_images)} pages: {pdf_path}")
        return generated_page_images, pdf_path

    def _sample_with_custom_style(
        self,
        lines: List[str],
        biases: List[float],
        custom_strokes: np.ndarray,
        custom_chars: str
    ) -> List[np.ndarray]:
        """
        Samples handwriting conditioned on the user's custom stroke sequence.
        """
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines]) if lines else 400

        x_prime = np.zeros([num_samples, 1200, 3], dtype=np.float32)
        x_prime_len = np.zeros([num_samples], dtype=np.int32)
        chars = np.zeros([num_samples, 120], dtype=np.int32)
        chars_len = np.zeros([num_samples], dtype=np.int32)

        primed_prefix = custom_chars[:35].strip()
        for i, line in enumerate(lines):
            # Ensure full string fits comfortably inside 120 char array
            max_line_len = max(10, 118 - len(primed_prefix) - 1)
            clipped_line = line[:max_line_len]
            full_str = f"{primed_prefix} {clipped_line}" if primed_prefix else clipped_line
            encoded = drawing.encode_ascii(full_str)[:120]
            encoded = np.array(encoded, dtype=np.int32)

            stroke_len = min(len(custom_strokes), 1200)
            x_prime[i, :stroke_len, :] = custom_strokes[:stroke_len]
            x_prime_len[i] = stroke_len
            chars[i, :len(encoded)] = encoded
            chars_len[i] = len(encoded)

        [samples] = self.hand.nn.session.run(
            [self.hand.nn.sampled_sequence],
            feed_dict={
                self.hand.nn.prime: True,
                self.hand.nn.x_prime: x_prime,
                self.hand.nn.x_prime_len: x_prime_len,
                self.hand.nn.num_samples: num_samples,
                self.hand.nn.sample_tsteps: max_tsteps,
                self.hand.nn.c: chars,
                self.hand.nn.c_len: chars_len,
                self.hand.nn.bias: biases
            }
        )
        return [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
