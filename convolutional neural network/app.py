"""Local browser UI: upload a digit or draw one with a mouse/touchscreen."""
from io import BytesIO
from pathlib import Path
import base64
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tensorflow import keras
from preprocess import prepare_image

st.set_page_config(page_title="MNIST CNN Digit Recognizer", page_icon="✍️")
st.title("Handwritten Digit Recognizer")
st.write("Upload one handwritten digit, or draw and download a PNG below, then upload it.")
st.caption("Runs locally. This model recognizes digits 0–9, not words or multi-digit numbers.")
# Use a standard HTML canvas, avoiding a version-sensitive drawing extension.
components.html("""
<canvas id="pad" width="280" height="280" style="border:1px solid #888;touch-action:none"></canvas><br>
<button id="clear">Clear</button> <button id="save">Download drawing</button>
<script>
const c=document.getElementById('pad'), x=c.getContext('2d');
function clear(){x.fillStyle='white';x.fillRect(0,0,280,280);}
clear(); x.lineWidth=20; x.lineCap='round'; x.strokeStyle='black';
let down=false;
function point(e){const r=c.getBoundingClientRect();return [
(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height];}
c.onpointerdown=e=>{down=true;c.setPointerCapture(e.pointerId);x.beginPath();x.moveTo(...point(e));};
c.onpointermove=e=>{if(down){x.lineTo(...point(e));x.stroke();}};
c.onpointerup=()=>down=false;c.onpointercancel=()=>down=false;
document.getElementById('clear').onclick=clear;
document.getElementById('save').onclick=()=>{const a=document.createElement('a');
a.download='my-digit.png';a.href=c.toDataURL('image/png');a.click();};
</script>
""", height=340)
model_path = Path(st.text_input("Model path", "artifacts/best_model.keras"))
invert = st.selectbox("Invert colors", ["auto", "yes", "no"])
upload = st.file_uploader("Upload a PNG, JPG, or WEBP containing one digit", type=["png", "jpg", "jpeg", "webp"])

@st.cache_resource
def load_model(path, modification_time):
    # The timestamp invalidates the cache after retraining at the same path.
    return keras.models.load_model(path)

if upload:
    try:
        with Image.open(BytesIO(upload.getvalue())) as image:
            st.image(image, caption="Original image", width=220)
            batch = prepare_image(image, invert=invert)
        st.image(batch[0, ..., 0], caption="Actual 28×28 model input", width=220, clamp=True)
        if not model_path.is_file():
            st.warning("Train a model first: python train.py")
        else:
            model = load_model(str(model_path), model_path.stat().st_mtime_ns)
            probabilities = model.predict(batch, verbose=0)[0]
            digit = int(probabilities.argmax())
            st.success(f"Prediction: {digit} — model probability: {probabilities[digit]:.1%}")
            st.bar_chart({"Probability": probabilities})
            st.caption("Probabilities are not calibrated certainty. Non-digit images may still receive confident predictions.")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
