# Rust Detector Mini 🔍

A small computer-vision tool that estimates the **percentage of rust (corrosion)** in an image of a pipe or metal
surface and classifies it as **Low / Medium / High**.

This is a lightweight, classical-CV version of my Smart India Hackathon (SIH) project on pipeline corrosion inspection.

## How it works

1. Blur the image to reduce noise
2. Convert from BGR to the **HSV** color space
3. **Threshold** the orange-brown / reddish-brown hue range that rust typically falls in (`cv2.inRange`)
4. Apply morphological opening and closing to remove small specks and fill small gaps
5. Rust coverage % = `rust pixels / total pixels * 100`
6. Classify: `< 5%` Low, `5-20%` Medium, `> 20%` High

## Tech stack

Python, OpenCV, NumPy, Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
rust_detector.py   # core logic (HSV thresholding + classification)
app.py             # Streamlit UI
requirements.txt
```

## Limitations

- Detection is color-based only, so brown paint or dirt can be mistaken for rust
- Saturation and brightness thresholds may need tuning under different lighting conditions

## Future improvements

- Train a CNN / segmentation model (e.g. U-Net) for more accurate detection
- Combine color with texture features to estimate severity
- Batch processing for multiple pipe images
