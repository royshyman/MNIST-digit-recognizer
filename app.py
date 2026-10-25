import gradio as gr
import numpy as np
import torch

from model import build_model
from preprocess import preprocess

MODEL_PATH = "mnist_mlp.pt"

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def to_grayscale(sketch):
    """Turn the Sketchpad value into a 2D uint8 array: white paper (255) with dark strokes."""
    # preprocess() crops to the ink, so the canvas size doesn't need to match anything
    if sketch is None or sketch.get("composite") is None:
        return None
    img = sketch["composite"].astype(np.float32)
    if img.ndim == 2:
        return img.astype(np.uint8)
    gray = img[..., :3].mean(axis=2)
    if img.shape[2] == 4:
        alpha = img[..., 3] / 255.0
        gray = gray * alpha + 255.0 * (1.0 - alpha)   # transparent pixels count as white paper
    return gray.astype(np.uint8)


def predict(sketch):
    gray = to_grayscale(sketch)
    if gray is None or gray.min() > 250:   # nothing drawn yet
        return {}, None
    x = preprocess(gray)                    # tensor (1, 1, 28, 28), normalized like the training data
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    # what the model sees, rescaled to 0-255 for display
    view = x[0, 0].numpy()
    view = (view - view.min()) / (view.max() - view.min() + 1e-8) * 255
    return {str(d): float(p) for d, p in enumerate(probs)}, view.astype(np.uint8)


CSS = """
body, gradio-app, .gradio-container { background: #1e1e1e !important; }
.gradio-container { width: 100%; max-width: 1120px !important; margin: 0 auto; }
#title h1 { color: #f2f2f2 !important; margin-bottom: 4px; }
#hint, #hint * { color: #b8b8b8 !important; }
#seen img { width: 224px !important; height: 224px !important; object-fit: contain; image-rendering: pixelated; }  /* enlarge the 28x28 image, keeping pixels crisp */
#canvas { background: #2c2c2c; border: 2px solid #555555; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5); }
#canvas button[aria-label="Brush Size"], #canvas button[aria-label="Color"] { display: none !important; }  /* one fixed brush size and colour */
"""

# The editor draws its whole area white with WebGL, so CSS can't reach it.
# Recolor the area around the drawable square once the editor has started.
JS = """
() => {
    const paint = () => {
        const app = globalThis.__PIXI_APP__;
        if (app && app.renderer) { app.renderer.background.color = 0x2c2c2c; }
        else { setTimeout(paint, 200); }
    };
    paint();

    // start with the brush already selected, so the user can draw right away
    const pickBrush = () => {
        const btn = document.querySelector('#canvas button[aria-label="Brush"]');
        if (btn) { btn.click(); }
        else { setTimeout(pickBrush, 200); }
    };
    pickBrush();
}
"""

with gr.Blocks(title="MNIST Digit Recognizer", fill_width=True) as demo:
    gr.Markdown("# Draw a digit", elem_id="title")
    gr.Markdown("Draw a single digit (0-9) in the white square. The prediction updates as you draw.", elem_id="hint")
    with gr.Row(equal_height=False):
        with gr.Column(scale=0, min_width=480):
            pad = gr.Sketchpad(
                type="numpy",
                image_mode="RGBA",
                canvas_size=(420, 420),
                height=500,
                width=480,
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=12),
                layers=False,
                transforms=(),
                label="Canvas",
                elem_id="canvas",
            )
        with gr.Column(scale=1, min_width=260):
            label = gr.Label(num_top_classes=3, label="Prediction")
            seen = gr.Image(
                label="What the model sees (28x28)",
                image_mode="L",
                height=260,
                interactive=False,
                buttons=[],
                elem_id="seen",
            )
    pad.change(predict, inputs=pad, outputs=[label, seen])

if __name__ == "__main__":
    demo.launch(css=CSS, js=JS)
