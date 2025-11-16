import json
import os
from typing import List, Sequence

import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.widgets import Button


def _is_point(candidate: Sequence) -> bool:
    """Return True if candidate looks like an [x, y] coordinate list."""
    return (
        isinstance(candidate, (list, tuple))
        and len(candidate) == 2
        and all(isinstance(coord, (int, float)) for coord in candidate)
    )


class ImageAnnotator:
    def __init__(self, image_dir: str, annotation_file: str = "annotation/annotations.json"):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.image_files = sorted(
            f for f in os.listdir(image_dir) if f.endswith((".nii.gz", ".png", ".jpg", ".jpeg"))
        )
        self.annotations = self.load_annotations()

        self.current_index = 0
        self.point_pairs: List[List[List[int]]] = []
        self.current_pair: List[List[int]] = []
        self.markers = []
        self.pair_colors = [
            "#FF6B6B",
            "#4C6EF5",
            "#40C057",
            "#F59F00",
            "#7950F2",
            "#2F9E44",
            "#D9480F",
            "#12B886",
            "#E67700",
            "#15AABF",
        ]

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.05, right=0.78, top=0.95, bottom=0.05)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.key_cid = self.fig.canvas.mpl_connect("key_press_event", self.on_keypress)
        button_width = 0.17
        button_height = 0.08
        button_left = 0.80
        button_top = 0.80
        button_spacing = 0.04

        button_palette = {
            "Next": ("#40C057", "#2F9E44"),
            "Previous": ("#4C6EF5", "#364FC7"),
            "Clear": ("#FF6B6B", "#FA5252"),
        }

        button_layout = [
            ("Next", self.next_image, "btn_next"),
            ("Previous", self.prev_image, "btn_prev"),
            ("Clear", self.clear_points, "btn_clear"),
        ]

        for idx, (label, handler, attr_name) in enumerate(button_layout):
            y_position = button_top - idx * (button_height + button_spacing)
            button_ax = plt.axes([button_left, y_position, button_width, button_height])
            color, hover = button_palette.get(label, ("#4C6EF5", "#364FC7"))
            button = Button(button_ax, label, color=color, hovercolor=hover)
            button.label.set_color("white")
            button.label.set_fontweight("bold")
            button.label.set_fontsize(10)
            button.on_clicked(handler)
            for spine in button_ax.spines.values():
                spine.set_visible(False)
            button_ax.set_xticks([])
            button_ax.set_yticks([])
            setattr(self, attr_name, button)

        self.find_next_unannotated()
        self.display_image()

    def load_annotations(self):
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_annotations(self):
        with open(self.annotation_file, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=4)

    def find_next_unannotated(self):
        unannotated_found = False
        for i, img_file in enumerate(self.image_files):
            if img_file not in self.annotations:
                self.current_index = i
                unannotated_found = True
                break

        if not unannotated_found:
            print("All images have been annotated. Loading first image for review.")
            self.current_index = 0

    def display_image(self):
        if self.current_index >= len(self.image_files):
            print("No more images to annotate.")
            plt.close(self.fig)
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        try:
            if image_path.endswith(".nii.gz"):
                itk_img = sitk.ReadImage(image_path)
                img = sitk.GetArrayFromImage(itk_img)
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)
            else:
                img = plt.imread(image_path)
        except Exception as exc:  # pragma: no cover - visual debugging
            print(f"Error loading image {image_path}: {exc}")
            self.ax.clear()
            self.ax.set_title(f"Error loading: {self.image_files[self.current_index]}")
            self.fig.canvas.draw_idle()
            return

        self.ax.clear()
        self.clear_markers()
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(
            f"Image: {self.image_files[self.current_index]} ({self.current_index + 1}/{len(self.image_files)})"
        )

        self.point_pairs = self.load_point_pairs_for_image(self.image_files[self.current_index])
        self.current_pair = []
        self.render_pairs()

    def onclick(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.current_pair.append([x, y])
        self.render_pairs()

        if len(self.current_pair) == 2:
            image_name = self.image_files[self.current_index]
            self.point_pairs.append([pt[:] for pt in self.current_pair])
            self.annotations[image_name] = [
                [[int(pt[0]), int(pt[1])] for pt in pair] for pair in self.point_pairs
            ]
            self.save_annotations()
            print(f"Saved pair #{len(self.point_pairs)} for {image_name}: {self.point_pairs[-1]}")
            self.current_pair = []
            self.render_pairs()

    def clear_points(self, event=None):
        self.current_pair = []
        self.point_pairs = []
        image_name = self.image_files[self.current_index]
        if image_name in self.annotations:
            del self.annotations[image_name]
            self.save_annotations()
        self.render_pairs()
        print("Cleared points for the current image.")

    def clear_markers(self):
        while self.markers:
            marker = self.markers.pop()
            if hasattr(marker, "remove") and marker.axes:
                marker.remove()

    def next_image(self, event):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
        else:
            self.find_next_unannotated()
        self.display_image()

    def prev_image(self, event):
        self.current_index = max(0, self.current_index - 1)
        self.display_image()

    def on_keypress(self, event):
        if event.key == "right":
            self.next_image(event)
        elif event.key == "left":
            self.prev_image(event)
        elif event.key == "escape":
            self.clear_points(event)

    def load_point_pairs_for_image(self, image_name: str) -> List[List[List[int]]]:
        entry = self.annotations.get(image_name)
        if not entry:
            return []
        normalized = self.normalize_entry(entry)
        normalized_pairs = []
        for pair in normalized:
            normalized_pairs.append([[int(point[0]), int(point[1])] for point in pair])
        return normalized_pairs

    @staticmethod
    def normalize_entry(entry):
        if ImageAnnotator.is_single_pair(entry):
            return [entry]

        normalized = []
        if isinstance(entry, list):
            for pair in entry:
                if ImageAnnotator.is_single_pair(pair):
                    normalized.append(pair)
        return normalized

    @staticmethod
    def is_single_pair(entry) -> bool:
        return isinstance(entry, list) and len(entry) == 2 and all(_is_point(point) for point in entry)

    def render_pairs(self):
        self.clear_markers()
        for idx, pair in enumerate(self.point_pairs):
            color = self.pair_colors[idx % len(self.pair_colors)]
            self.draw_pair(pair, color=color, label=str(idx + 1))
        if self.current_pair:
            color = self.pair_colors[len(self.point_pairs) % len(self.pair_colors)]
            self.draw_pair(self.current_pair, color=color)
        self.fig.canvas.draw_idle()

    def draw_pair(self, points: List[List[int]], color: str, label: str | None = None):
        if len(points) >= 2:
            xs = [pt[0] for pt in points[:2]]
            ys = [pt[1] for pt in points[:2]]
            line, = self.ax.plot(xs, ys, linestyle="--", color=color, linewidth=1)
            self.markers.append(line)
        for x, y in points:
            marker, = self.ax.plot(x, y, marker="o", color=color, markersize=4)
            self.markers.append(marker)
        if label and points:
            label_x, label_y = points[0]
            text = self.ax.text(
                label_x + 5,
                label_y + 5,
                label,
                color=color,
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            )
            self.markers.append(text)


def main():
    image_directory = "annotation/labelsTr"
    annotator = ImageAnnotator(image_directory)
    plt.show()


if __name__ == "__main__":
    main()

