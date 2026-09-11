"""Gradio web app for chest X-ray disease classification.

Loads the saved best model and per-class F1 thresholds, then serves a small UI
where an uploaded X-ray is classified. Every disease whose probability is above
its threshold is reported with its probability; if none pass, the app reports
"No Finding".

Run:
    python app.py
"""
import argparse
import os

import gradio as gr

from predictor import load_artifacts, predict_image

DEFAULT_MODEL_PATH = os.path.join("chest_xray_run", "run2_best.pth")
DEFAULT_THRESHOLDS_PATH = os.path.join("chest_xray_run", "f1_thresholds.npy")


def build_demo(model, thresholds, device):
    def predict(image):
        if image is None:
            return "Please upload a chest X-ray image.", {}
        results = predict_image(image, model, thresholds, device)
        if not results:
            return "No Finding (no disease detected)", {}
        lines = [f"{disease}: {prob * 100:.2f}%" for disease, prob in
                 sorted(results.items(), key=lambda kv: kv[1], reverse=True)]
        return "\n".join(lines), results

    with gr.Blocks(title="Chest X-Ray Disease Classification") as demo:
        gr.Markdown(
            "# Chest X-Ray Disease Classification\n"
            "Upload a frontal chest X-ray. The model returns every disease whose "
            "probability exceeds its tuned threshold, or **No Finding** if none do.")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Chest X-ray", type="pil")
                button = gr.Button("Classify")
            with gr.Column():
                text_out = gr.Textbox(label="Prediction")
                json_out = gr.JSON(label="Diseases above threshold (probabilities)")

        button.click(predict, inputs=input_image, outputs=[text_out, json_out])
        input_image.change(predict, inputs=input_image, outputs=[text_out, json_out])
    return demo


def main(model_path=DEFAULT_MODEL_PATH, thresholds_path=DEFAULT_THRESHOLDS_PATH, server_port=7860, launch=True):
    print("Loading model and thresholds ...")
    model, thresholds, device = load_artifacts(model_path, thresholds_path)
    print(f"Loaded. Running on {device}.")

    demo = build_demo(model, thresholds, device)
    if launch:
        demo.launch(server_name="127.0.0.1", server_port=server_port)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--thresholds-path", default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--no-launch", action="store_true", help="Build the UI and exit without serving.")
    args = parser.parse_args()
    main(args.model_path, args.thresholds_path, args.server_port, launch=not args.no_launch)