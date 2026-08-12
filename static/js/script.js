/**
 * script.js
 * ---------
 * Handles all client-side interaction for the Speech Emotion Detection app:
 *   - File selection (click or drag-and-drop)
 *   - Local audio preview
 *   - Sending the file to the Flask /predict endpoint
 *   - Rendering the predicted emotion, emoji, and confidence score
 */

(function () {
    "use strict";

    const dropzone = document.getElementById("dropzone");
    const audioInput = document.getElementById("audioInput");
    const fileNameLabel = document.getElementById("fileNameLabel");
    const audioPreview = document.getElementById("audioPreview");
    const detectBtn = document.getElementById("detectBtn");
    const detectBtnText = document.getElementById("detectBtnText");
    const detectSpinner = document.getElementById("detectSpinner");
    const errorBox = document.getElementById("errorBox");

    const resultPanel = document.getElementById("resultPanel");
    const resultEmoji = document.getElementById("resultEmoji");
    const resultEmotion = document.getElementById("resultEmotion");
    const resultFilename = document.getElementById("resultFilename");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceFill = document.getElementById("confidenceFill");
    const scoreBreakdown = document.getElementById("scoreBreakdown");

    let selectedFile = null;

    /** Show a file the user picked (via click or drag-drop) and enable the button. */
    function handleFileSelected(file) {
        if (!file) return;

        selectedFile = file;
        fileNameLabel.textContent = file.name;

        const objectUrl = URL.createObjectURL(file);
        audioPreview.src = objectUrl;
        audioPreview.classList.remove("d-none");

        detectBtn.disabled = false;
        hideError();
        resultPanel.classList.add("d-none");
    }

    function showError(message) {
        errorBox.textContent = message;
        errorBox.classList.remove("d-none");
    }

    function hideError() {
        errorBox.classList.add("d-none");
        errorBox.textContent = "";
    }

    function setLoading(isLoading) {
        detectBtn.disabled = isLoading || !selectedFile;
        detectSpinner.classList.toggle("d-none", !isLoading);
        detectBtnText.textContent = isLoading ? "Analyzing..." : "Detect Emotion";
    }

    function renderResult(data) {
        resultEmoji.textContent = data.emoji;
        resultEmotion.textContent = data.emotion;
        resultFilename.textContent = data.filename;
        confidenceValue.textContent = `${data.confidence}%`;

        // Animate the confidence bar from 0 to its target width.
        confidenceFill.style.width = "0%";
        requestAnimationFrame(() => {
            confidenceFill.style.width = `${data.confidence}%`;
        });

        // Render the full per-class score breakdown, sorted highest first.
        scoreBreakdown.innerHTML = "";
        const entries = Object.entries(data.all_scores || {})
            .sort((a, b) => b[1] - a[1]);

        for (const [label, score] of entries) {
            const row = document.createElement("div");
            row.className = "score-row";
            row.innerHTML = `<span>${capitalize(label)}</span><span>${score}%</span>`;
            scoreBreakdown.appendChild(row);
        }

        resultPanel.classList.remove("d-none");
        resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function capitalize(word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
    }

    async function detectEmotion() {
        if (!selectedFile) return;

        hideError();
        setLoading(true);

        const formData = new FormData();
        formData.append("audio_file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Something went wrong while analyzing the audio.");
            }

            renderResult(data);
        } catch (err) {
            showError(err.message);
            resultPanel.classList.add("d-none");
        } finally {
            setLoading(false);
        }
    }

    // -- Event wiring --------------------------------------------------

    audioInput.addEventListener("change", (event) => {
        handleFileSelected(event.target.files[0]);
    });

    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
        const file = event.dataTransfer.files[0];
        if (file) {
            audioInput.files = event.dataTransfer.files;
            handleFileSelected(file);
        }
    });

    detectBtn.addEventListener("click", detectEmotion);
})();
