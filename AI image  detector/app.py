from flask import Flask, request, render_template_string, url_for
from PIL import Image, ExifTags
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import os
import uuid
import urllib.parse

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_NAME = "Smogy/SMOGY-Ai-images-detector"

print("Loading AI detection model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully.")


def get_metadata(image):

    metadata = []

    try:
        exif = image.getexif()

        for key, value in exif.items():

            name = ExifTags.TAGS.get(key, str(key))
            value_text = str(value)

            if any(word in name.lower() for word in [
                "software",
                "description",
                "comment",
                "maker",
                "model"
            ]):

                metadata.append(
                    name + ": " + value_text
                )

    except Exception:
        pass

    try:

        for key, value in image.info.items():

            value_text = str(value)

            if any(word in key.lower() for word in [
                "prompt",
                "negative",
                "parameters",
                "workflow",
                "software",
                "generator",
                "comment",
                "model"
            ]):

                metadata.append(
                    key + ": " + value_text
                )

    except Exception:
        pass

    return " ".join(metadata)


def detect_prompt_metadata(image):

    metadata = get_metadata(image).lower()

    ai_tools = [
        "prompt",
        "negative prompt",
        "stable diffusion",
        "automatic1111",
        "comfyui",
        "midjourney",
        "dall-e",
        "dall·e",
        "openai",
        "leonardo ai",
        "adobe firefly",
        "flux",
        "novelai",
        "invokeai",
        "fooocus",
        "dreamstudio"
    ]

    detected = []

    for tool in ai_tools:

        if tool in metadata:

            detected.append(tool)

    return list(set(detected))


def analyze_image(image):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    labels = model.config.id2label

    ai_score = 0
    real_score = 0

    for index, probability in enumerate(probabilities):

        label = str(labels[index]).lower()

        if any(word in label for word in [
            "artificial",
            "ai",
            "generated",
            "fake"
        ]):

            ai_score += probability.item()

        elif any(word in label for word in [
            "human",
            "real",
            "photograph"
        ]):

            real_score += probability.item()

    total = ai_score + real_score

    if total > 0:

        ai_score = ai_score / total
        real_score = real_score / total

    return ai_score * 100, real_score * 100


def create_share_links(result, ai, real):

    text = (
        "AI Image Detector Result: "
        + result
        + " | AI: "
        + str(round(ai, 2))
        + "% | Real: "
        + str(round(real, 2))
        + "%"
    )

    encoded = urllib.parse.quote(text)

    whatsapp = (
        "https://wa.me/?text=" + encoded
    )

    telegram = (
        "https://t.me/share/url?text=" + encoded
    )

    linkedin = (
        "https://www.linkedin.com/sharing/share-offsite/"
    )

    instagram = (
        "https://www.instagram.com/"
    )

    return whatsapp, telegram, linkedin, instagram


HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>VisionGuard AI</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
    Arial,
    Helvetica,
    sans-serif;

    color: white;

    background:

    radial-gradient(
        circle at 10% 10%,
        rgba(239,68,68,0.20),
        transparent 30%
    ),

    radial-gradient(
        circle at 90% 20%,
        rgba(127,29,29,0.18),
        transparent 30%
    ),

    #050505;
}


.navbar {

    width: 92%;

    margin: 20px auto;

    padding: 18px 25px;

    display: flex;

    justify-content: space-between;

    align-items: center;

    border-radius: 18px;

    border: 1px solid #3b1111;

    background:
    linear-gradient(
        135deg,
        #181818,
        #090909
    );

    box-shadow:
    0 15px 40px rgba(0,0,0,.4);
}


.logo {

    font-size: 23px;

    font-weight: 900;
}

.logo span {

    color: #ef4444;
}


.nav {

    display: flex;

    gap: 28px;

    color: #a1a1aa;

    font-size: 13px;
}


.hero {

    width: 92%;

    margin: 25px auto;

    text-align: center;

    padding: 65px 25px;

    border-radius: 28px;

    border: 1px solid #541414;

    background:

    linear-gradient(
        135deg,
        #3a0c0c,
        #111111
    );

    box-shadow:
    0 30px 70px rgba(0,0,0,.45);
}


.badge {

    display: inline-block;

    padding: 9px 18px;

    border-radius: 30px;

    background: #240808;

    border: 1px solid #7f1d1d;

    color: #fca5a5;

    font-size: 12px;

    letter-spacing: 1.5px;
}


.hero h1 {

    font-size: 50px;

    margin: 22px 0 10px;

    font-weight: 900;
}


.hero h1 span {

    color: #ef4444;
}


.hero p {

    max-width: 720px;

    margin: auto;

    color: #a1a1aa;

    line-height: 1.7;

    font-size: 16px;
}


.features {

    width: 92%;

    margin: 25px auto;

    display: grid;

    grid-template-columns:
    repeat(3,1fr);

    gap: 15px;
}


.feature {

    padding: 25px;

    text-align: center;

    border-radius: 18px;

    border: 1px solid #321313;

    background: #101010;
}


.feature-icon {

    font-size: 35px;
}


.feature h3 {

    margin: 10px 0;

    color: white;
}


.feature p {

    color: #71717a;

    font-size: 13px;
}


.container {

    width: 92%;

    margin: 30px auto;

    display: grid;

    grid-template-columns:
    1fr 1fr;

    gap: 20px;
}


.panel {

    background:
    linear-gradient(
        145deg,
        #171717,
        #090909
    );

    border: 1px solid #3a1414;

    border-radius: 24px;

    padding: 25px;

    box-shadow:
    0 20px 50px rgba(0,0,0,.4);
}


.panel h2 {

    margin-top: 0;

    font-size: 20px;
}


.upload-box {

    border: 2px dashed #6b1b1b;

    border-radius: 18px;

    padding: 35px 20px;

    text-align: center;

    background: #0d0d0d;

    transition: .2s;
}


.upload-box:hover {

    border-color: #ef4444;

    background: #160909;
}


input[type=file] {

    width: 100%;

    padding: 15px;

    color: #aaa;

    background: #111;

    border-radius: 10px;

    border: 1px solid #333;
}


.button {

    display: block;

    width: 100%;

    margin-top: 20px;

    padding: 15px;

    border: none;

    border-radius: 12px;

    background:
    linear-gradient(
        90deg,
        #991b1b,
        #ef4444
    );

    color: white;

    font-size: 16px;

    font-weight: 800;

    cursor: pointer;
}


.button:hover {

    filter: brightness(1.15);
}


.preview {

    width: 100%;

    max-height: 400px;

    object-fit: contain;

    border-radius: 15px;

    margin-top: 20px;

    background: #050505;
}


.result {

    text-align: center;

    padding: 10px;
}


.result-icon {

    font-size: 60px;
}


.result-title {

    font-size: 30px;

    font-weight: 900;

    margin: 10px 0;
}


.ai {

    color: #f87171;
}


.real {

    color: #4ade80;
}


.prompt {

    color: #fbbf24;
}


.description {

    color: #a1a1aa;

    font-size: 14px;

    line-height: 1.6;
}


.confidence {

    margin-top: 20px;

    padding: 22px;

    background: #080808;

    border: 1px solid #321313;

    border-radius: 18px;
}


.confidence-label {

    color: #71717a;

    font-size: 11px;

    letter-spacing: 2px;
}


.confidence-number {

    font-size: 42px;

    font-weight: 900;

    margin-top: 7px;
}


.score {

    text-align: left;

    margin-top: 25px;
}


.score-header {

    display: flex;

    justify-content: space-between;

    margin-bottom: 8px;

    font-size: 14px;
}


.bar {

    height: 13px;

    background: #27272a;

    border-radius: 20px;

    overflow: hidden;
}


.ai-bar {

    height: 100%;

    background:
    linear-gradient(
        90deg,
        #991b1b,
        #ef4444
    );
}


.real-bar {

    height: 100%;

    background:
    linear-gradient(
        90deg,
        #15803d,
        #22c55e
    );
}


.metadata {

    margin-top: 25px;

    padding: 18px;

    background: #17120a;

    border: 1px solid #5b4212;

    border-radius: 15px;

    text-align: left;
}


.metadata h3 {

    margin-top: 0;

    color: #fbbf24;
}


.metadata p {

    color: #a1a1aa;

    font-size: 13px;
}


.share {

    margin-top: 25px;

    padding: 20px;

    background: #101010;

    border: 1px solid #321313;

    border-radius: 16px;
}


.share h3 {

    margin-top: 0;

    text-align: center;
}


.share-buttons {

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

    justify-content: center;
}


.share-button {

    text-decoration: none;

    color: white;

    padding: 11px 16px;

    border-radius: 10px;

    font-size: 13px;

    font-weight: bold;
}


.whatsapp {

    background: #075e54;
}


.telegram {

    background: #168acd;
}


.linkedin {

    background: #0a66c2;
}


.instagram {

    background:
    linear-gradient(
        45deg,
        #f59e0b,
        #ec4899,
        #8b5cf6
    );
}


.warning {

    margin-top: 20px;

    padding: 13px;

    border-radius: 12px;

    background: #1c1111;

    border: 1px solid #451a1a;

    color: #a1a1aa;

    font-size: 11px;

    line-height: 1.5;
}


.footer {

    width: 92%;

    margin: 30px auto;

    text-align: center;

    padding: 25px;

    color: #52525b;

    font-size: 12px;
}


@media(max-width:800px) {

    .container {

        grid-template-columns: 1fr;
    }

    .features {

        grid-template-columns: 1fr;
    }

    .nav {

        display: none;
    }

    .hero h1 {

        font-size: 35px;
    }
}

</style>

</head>


<body>


<div class="navbar">

    <div class="logo">
        Vision<span>Guard</span> AI
    </div>

    <div class="nav">

        <span>HOME</span>
        <span>DETECTION</span>
        <span>ANALYSIS</span>
        <span>ABOUT</span>

    </div>

</div>


<div class="hero">

    <div class="badge">
        ● DEEP LEARNING IMAGE INTELLIGENCE
    </div>

    <h1>
        Detect <span>AI Images</span><br>
        With Deep Learning
    </h1>

    <p>
        Upload an image and analyze whether it is
        AI-generated, real photography, or contains
        recognizable AI-generation metadata.
    </p>

</div>


<div class="features">

    <div class="feature">

        <div class="feature-icon">
            🤖
        </div>

        <h3>
            AI Detection
        </h3>

        <p>
            Deep-learning visual analysis
            for AI-generated images.
        </p>

    </div>


    <div class="feature">

        <div class="feature-icon">
            ✨
        </div>

        <h3>
            Prompt Detection
        </h3>

        <p>
            Checks available metadata for
            recognizable AI prompt information.
        </p>

    </div>


    <div class="feature">

        <div class="feature-icon">
            📊
        </div>

        <h3>
            Confidence
        </h3>

        <p>
            View AI and real photography
            prediction percentages.
        </p>

    </div>

</div>


<div class="container">


<div class="panel">

    <h2>
        📤 Upload Image
    </h2>

    <div class="upload-box">

        <p>
            Select an image to analyze
        </p>

        <form
            method="POST"
            enctype="multipart/form-data"
        >

            <input
                type="file"
                name="image"
                accept="image/*"
                required
            >

            <button
                class="button"
                type="submit"
            >
                🔴 ANALYZE IMAGE
            </button>

        </form>

    </div>


    {% if image_url %}

        <img
            src="{{ image_url }}"
            class="preview"
        >

    {% endif %}

</div>


<div class="panel">

    <h2>
        🧠 Detection Result
    </h2>


    {% if result %}

    <div class="result">

        <div class="result-icon">
            {{ icon }}
        </div>

        <div class="result-title {{ result_class }}">
            {{ result }}
        </div>

        <div class="description">
            {{ description }}
        </div>


        <div class="confidence">

            <div class="confidence-label">
                DETECTION CONFIDENCE
            </div>

            <div class="confidence-number">
                {{ confidence }}%
            </div>

        </div>


        <div class="score">

            <div class="score-header">

                <span>
                    🤖 AI Generated
                </span>

                <strong>
                    {{ ai }}%
                </strong>

            </div>

            <div class="bar">

                <div
                    class="ai-bar"
                    style="width:{{ ai }}%"
                ></div>

            </div>

        </div>


        <div class="score">

            <div class="score-header">

                <span>
                    📷 Real Photography
                </span>

                <strong>
                    {{ real }}%
                </strong>

            </div>

            <div class="bar">

                <div
                    class="real-bar"
                    style="width:{{ real }}%"
                ></div>

            </div>

        </div>


        <div class="metadata">

            <h3>
                ✨ AI Prompt Analysis
            </h3>

            <p>
                {{ metadata }}
            </p>

        </div>


        <div class="share">

            <h3>
                SHARE RESULT
            </h3>

            <div class="share-buttons">

                <a
                    class="share-button whatsapp"
                    href="{{ whatsapp }}"
                    target="_blank"
                >
                    WhatsApp
                </a>

                <a
                    class="share-button telegram"
                    href="{{ telegram }}"
                    target="_blank"
                >
                    Telegram
                </a>

                <a
                    class="share-button linkedin"
                    href="{{ linkedin }}"
                    target="_blank"
                >
                    LinkedIn
                </a>

                <a
                    class="share-button instagram"
                    href="{{ instagram }}"
                    target="_blank"
                >
                    Instagram
                </a>

            </div>

        </div>


        <div class="warning">

            ⚠ This result is a deep-learning
            model prediction and is not absolute proof.
            Image compression, editing, resizing and
            unseen AI generators can affect detection.

        </div>

    </div>

    {% else %}

    <div
        style="
        text-align:center;
        padding:120px 20px;
        color:#71717a;
        "
    >

        <div style="font-size:55px;">
            🔍
        </div>

        <h2 style="color:#d4d4d8;">
            Ready to Analyze
        </h2>

        <p>
            Upload an image and click
            Analyze Image.
        </p>

    </div>

    {% endif %}

</div>


</div>


<div class="footer">

    <strong>
        VisionGuard AI
    </strong>

    <br>

    Deep Learning Mini Project
    • AI Image Detection System

</div>


</body>

</html>

"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    icon = ""
    result_class = ""
    description = ""
    confidence = 0
    ai = 0
    real = 0
    metadata = ""
    image_url = None

    whatsapp = "#"
    telegram = "#"
    linkedin = "#"
    instagram = "https://www.instagram.com/"

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            extension = os.path.splitext(
                file.filename
            )[1].lower()

            allowed = [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]

            if extension not in allowed:

                return "Unsupported image format."

            filename = (
                str(uuid.uuid4())
                + extension
            )

            path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(path)

            image = Image.open(path).convert("RGB")

            image_url = url_for(
                "uploaded_file",
                filename=filename
            )

            ai, real = analyze_image(image)

            prompt_matches = detect_prompt_metadata(
                Image.open(path)
            )

            if prompt_matches:

                result = "AI PROMPT IMAGE"

                icon = "✨"

                result_class = "prompt"

                confidence = ai

                description = (
                    "Recognizable AI-generation or "
                    "prompt-related metadata was found "
                    "inside the image."
                )

                metadata = (
                    "Detected indicators: "
                    + ", ".join(prompt_matches)
                )

            elif ai >= real:

                result = "AI-GENERATED IMAGE"

                icon = "🤖"

                result_class = "ai"

                confidence = ai

                description = (
                    "The deep-learning model found "
                    "stronger visual characteristics "
                    "associated with AI-generated content."
                )

                metadata = (
                    "No recognizable AI prompt metadata "
                    "was found."
                )

            else:

                result = "REAL PHOTOGRAPHY"

                icon = "📷"

                result_class = "real"

                confidence = real

                description = (
                    "The deep-learning model found "
                    "stronger characteristics associated "
                    "with real photography."
                )

                metadata = (
                    "No recognizable AI prompt metadata "
                    "was found."
                )

            whatsapp, telegram, linkedin, instagram = (
                create_share_links(
                    result,
                    ai,
                    real
                )
            )

    return render_template_string(
        HTML,
        result=result,
        icon=icon,
        result_class=result_class,
        description=description,
        confidence=round(confidence, 2),
        ai=round(ai, 2),
        real=round(real, 2),
        metadata=metadata,
        image_url=image_url,
        whatsapp=whatsapp,
        telegram=telegram,
        linkedin=linkedin,
        instagram=instagram
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":

    print("")
    print("======================================")
    print("       VISIONGUARD AI DETECTOR")
    print("======================================")
    print("")
    print("Open this in your browser:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )