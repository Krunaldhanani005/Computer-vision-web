// ── Identity Engine — training helpers ───────────────────────────────────────
let images = [];
let activeFeature = null;  // 'face' | 'emotion' | 'object' | null

function handleUpload(files) {
    for (let file of files) images.push(file);
    renderImages();
}

function renderImages() {
    const container = document.getElementById("preview");
    if (!container) return;
    container.innerHTML = "";
    images.forEach((file, index) => {
        const url = URL.createObjectURL(file);
        container.innerHTML += `
        <div class="img-wrapper">
            <img src="${url}" width="70">
            <button class="remove-btn" onclick="event.stopPropagation(); removeImage(${index})">✖</button>
        </div>`;
    });
}

function removeImage(index) {
    images.splice(index, 1);
    renderImages();
}

function setExternalStatus(msg, type) {
    const el = document.getElementById('global-status');
    if (el) { el.textContent = msg; el.className = 'status-box ' + type; }
}

function trainModel() {
    const nameElem = document.getElementById("name");
    if (!nameElem) return;
    const name = nameElem.value.trim();

    if (!name || images.length === 0) {
        setExternalStatus("Enter name and upload images", "error");
        setTimeout(() => setExternalStatus('Ready', ''), 2000);
        return;
    }

    setExternalStatus('Uploading and communicating with AI module...', 'info');

    const formData = new FormData();
    formData.append("name", name);
    images.forEach(img => formData.append("images", img));

    fetch("/train", { method: "POST", body: formData })
        .then(async res => ({ text: await res.text(), ok: res.ok }))
        .then(data => {
            if (!data.ok) {
                setExternalStatus("Error: " + data.text, "error");
                setTimeout(() => setExternalStatus('Ready', ''), 4000);
            } else {
                setExternalStatus("Training Complete ✅", "success");
                images = [];
                document.getElementById("name").value = "";
                renderImages();
                setTimeout(() => setExternalStatus('Ready', ''), 3000);
            }
        })
        .catch(err => {
            console.error(err);
            setExternalStatus("Error connecting to server.", "error");
            setTimeout(() => setExternalStatus('Ready', ''), 4000);
        });
}

function cancelTraining() {
    images = [];
    const nameElem = document.getElementById("name");
    const fileInput = document.getElementById("fileInput");
    if (nameElem) nameElem.value = "";
    if (fileInput) fileInput.value = "";
    renderImages();
    setExternalStatus('Clearing training memory...', 'info');

    fetch("/clear_training", { method: "POST" })
        .then(() => {
            setExternalStatus('Training cleared', 'info');
            setTimeout(() => setExternalStatus('Ready', ''), 2000);
        })
        .catch(() => {
            setExternalStatus('Session cache reset', 'info');
            setTimeout(() => setExternalStatus('Ready', ''), 2000);
        });
}

// ── DOM Ready — camera controls ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    // ── Identity Engine controls ──────────────────────────────────────────────
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const videoFeed = document.getElementById('video-feed');
    const placeholder = document.getElementById('video-placeholder');
    const indicator = document.getElementById('live-indicator');
    const globalStatus = document.getElementById('global-status');

    function setStatus(msg, type) {
        if (!globalStatus) return;
        globalStatus.textContent = msg;
        globalStatus.className = 'status-box ' + type;
    }

    function resetFaceUI() {
        if (videoFeed) { videoFeed.src = ''; videoFeed.style.display = 'none'; }
        if (placeholder) placeholder.style.display = 'flex';
        if (indicator) { indicator.classList.remove('active'); indicator.style.color = 'var(--text-dim)'; indicator.textContent = 'ACTIVE'; }
        if (startBtn) startBtn.style.display = 'flex';
        if (stopBtn) stopBtn.style.display = 'none';
        setStatus('Ready');
    }


    // ── Emotion Engine controls ───────────────────────────────────────────────
    const emotionStartBtn = document.getElementById('emotion-start-btn');
    const emotionStopBtn = document.getElementById('emotion-stop-btn');
    const emotionFeed = document.getElementById('emotion-feed');
    const emotionPlaceholder = document.getElementById('emotion-placeholder');
    const emotionIndicator = document.getElementById('emotion-indicator');
    const emotionStatus = document.getElementById('emotion-status');

    function setEmotionStatus(msg, type) {
        if (!emotionStatus) return;
        emotionStatus.textContent = msg;
        emotionStatus.className = 'status-box ' + type;
    }

    function resetEmotionUI() {
        if (emotionFeed) { emotionFeed.src = ''; emotionFeed.style.display = 'none'; }
        if (emotionPlaceholder) emotionPlaceholder.style.display = 'flex';
        if (emotionIndicator) { emotionIndicator.classList.remove('active'); emotionIndicator.style.color = 'var(--text-dim)'; emotionIndicator.textContent = 'READY'; }
        if (emotionStartBtn) emotionStartBtn.style.display = 'flex';
        if (emotionStopBtn) emotionStopBtn.style.display = 'none';
        setEmotionStatus('Ready');
    }

    // ── Object Detection controls ─────────────────────────────────────────────
    const objectStartBtn = document.getElementById('object-start-btn');
    const objectStopBtn = document.getElementById('object-stop-btn');
    const objectFeed = document.getElementById('object-feed');
    const objectPlaceholder = document.getElementById('object-placeholder');
    const objectIndicator = document.getElementById('object-indicator');
    const objectStatus = document.getElementById('object-status');

    function setObjectStatus(msg, type) {
        if (!objectStatus) return;
        objectStatus.textContent = msg;
        objectStatus.className = 'status-box ' + type;
    }

    function resetObjectUI() {
        if (objectFeed) { objectFeed.src = ''; objectFeed.style.display = 'none'; }
        if (objectPlaceholder) objectPlaceholder.style.display = 'flex';
        if (objectIndicator) { objectIndicator.classList.remove('active'); objectIndicator.style.color = 'var(--text-dim)'; objectIndicator.textContent = 'READY'; }
        if (objectStartBtn) objectStartBtn.style.display = 'flex';
        if (objectStopBtn) objectStopBtn.style.display = 'none';
        setObjectStatus('Ready');
    }

    // ── Feature isolation — only one stream at a time ─────────────────────────
    async function stopAllFeatures() {
        if (activeFeature === 'face') {
            await fetch('/stop_camera', { method: 'POST' }).catch(() => { });
            resetFaceUI();
        } else if (activeFeature === 'emotion') {
            await fetch('/stop_emotion_camera', { method: 'POST' }).catch(() => { });
            resetEmotionUI();
        } else if (activeFeature === 'object') {
            await fetch('/stop_object_camera', { method: 'POST' }).catch(() => { });
            resetObjectUI();
        }
        activeFeature = null;
    }

    // ── Face Recognition — event listeners ───────────────────────────────────
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            await stopAllFeatures();
            setStatus('Starting webcam...', 'info');
            try {
                const resp = await fetch('/start_camera', { method: 'POST' });
                const result = await resp.json();
                if (result.success) {
                    activeFeature = 'face';
                    if (placeholder) placeholder.style.display = 'none';
                    if (videoFeed) { videoFeed.style.display = 'block'; videoFeed.src = `/video_feed?t=${Date.now()}`; }
                    if (indicator) { indicator.classList.add('active'); indicator.style.color = 'var(--success)'; indicator.textContent = 'DETECTING...'; }
                    if (startBtn) startBtn.style.display = 'none';
                    if (stopBtn) stopBtn.style.display = 'flex';
                    setStatus('AI Recognition Active.', 'success');
                } else {
                    setStatus(result.message, 'error');
                }
            } catch (err) {
                setStatus('Could not start camera.', 'error');
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            try {
                await fetch('/stop_camera', { method: 'POST' });
                activeFeature = null;
                resetFaceUI();
                setStatus('Camera stopped.', 'info');
                setTimeout(() => setStatus('Ready', ''), 3000);
            } catch (err) {
                setStatus('Error stopping camera.', 'error');
            }
        });
    }


    // ── Emotion Detection — event listeners ───────────────────────────────────
    if (emotionStartBtn) {
        emotionStartBtn.addEventListener('click', async () => {
            await stopAllFeatures();
            setEmotionStatus('Starting emotion detection...', 'info');
            try {
                const resp = await fetch('/start_emotion_camera', { method: 'POST' });
                const result = await resp.json();
                if (result.success) {
                    activeFeature = 'emotion';
                    if (emotionPlaceholder) emotionPlaceholder.style.display = 'none';
                    if (emotionFeed) { emotionFeed.style.display = 'block'; emotionFeed.src = `/start_emotion?t=${Date.now()}`; }
                    if (emotionIndicator) { emotionIndicator.classList.add('active'); emotionIndicator.style.color = 'var(--emotion)'; emotionIndicator.textContent = 'ANALYSING...'; }
                    if (emotionStartBtn) emotionStartBtn.style.display = 'none';
                    if (emotionStopBtn) emotionStopBtn.style.display = 'flex';
                    setEmotionStatus('Emotion Detection Active.', 'emotion');
                } else {
                    setEmotionStatus(result.message || 'Could not start camera.', 'error');
                }
            } catch (err) {
                setEmotionStatus('Could not start emotion camera.', 'error');
            }
        });
    }

    if (emotionStopBtn) {
        emotionStopBtn.addEventListener('click', async () => {
            try {
                await fetch('/stop_emotion_camera', { method: 'POST' });
                activeFeature = null;
                resetEmotionUI();
                setEmotionStatus('Emotion detection stopped.', 'info');
                setTimeout(() => setEmotionStatus('Ready', ''), 3000);
            } catch (err) {
                setEmotionStatus('Error stopping camera.', 'error');
            }
        });
    }

    // ── Object Detection — event listeners ────────────────────────────────────
    if (objectStartBtn) {
        objectStartBtn.addEventListener('click', async () => {
            await stopAllFeatures();
            setObjectStatus('Starting object detection...', 'info');
            try {
                const resp = await fetch('/start_object_camera', { method: 'POST' });
                const result = await resp.json();
                if (result.success) {
                    activeFeature = 'object';
                    if (objectPlaceholder) objectPlaceholder.style.display = 'none';
                    if (objectFeed) { objectFeed.style.display = 'block'; objectFeed.src = `/start_object_detection?t=${Date.now()}`; }
                    if (objectIndicator) { objectIndicator.classList.add('active'); objectIndicator.style.color = 'var(--object)'; objectIndicator.textContent = 'SCANNING...'; }
                    if (objectStartBtn) objectStartBtn.style.display = 'none';
                    if (objectStopBtn) objectStopBtn.style.display = 'flex';
                    setObjectStatus('Object Detection Active.', 'object');
                } else {
                    setObjectStatus(result.message || 'Could not start camera.', 'error');
                }
            } catch (err) {
                setObjectStatus('Could not start object detection.', 'error');
            }
        });
    }

    if (objectStopBtn) {
        objectStopBtn.addEventListener('click', async () => {
            try {
                await fetch('/stop_object_camera', { method: 'POST' });
                activeFeature = null;
                resetObjectUI();
                setObjectStatus('Object detection stopped.', 'info');
                setTimeout(() => setObjectStatus('Ready', ''), 3000);
            } catch (err) {
                setObjectStatus('Error stopping camera.', 'error');
            }
        });
    }

    // FIXED VARIABLES (NO DUPLICATES)
    const vehiclePlaceholder = document.getElementById('vehicle-placeholder');
    const videoFeedVehicle = document.getElementById('videoFeed'); // renamed to avoid conflict
    const vehicleStatus = document.getElementById('vehicle-status');
    const uploadBtnText = document.getElementById('upload-btn-text');
    const uploadSpinner = document.getElementById('upload-spinner');

    function setVehicleStatus(msg, type) {
        if (!vehicleStatus) return;
        vehicleStatus.textContent = msg;
        vehicleStatus.className = 'status-box ' + type;
    }

    const videoInput = document.getElementById("videoInput");
    const uploadBtnVehicle = document.getElementById("vehicle-upload-btn");
    const uploadFormVehicle = document.getElementById("uploadForm"); // renamed

    // ENABLE BUTTON AFTER SELECT
    if (videoInput && uploadBtnVehicle) {
        videoInput.addEventListener("change", () => {
            if (videoInput.files.length > 0) {
                uploadBtnVehicle.disabled = false;
                uploadBtnVehicle.style.opacity = '1';
                uploadBtnVehicle.style.cursor = 'pointer';
            } else {
                uploadBtnVehicle.disabled = true;
                uploadBtnVehicle.style.opacity = '0.5';
                uploadBtnVehicle.style.cursor = 'not-allowed';
            }
        });
    }

    let isVehicleRunning = false;

    // UPLOAD HANDLER
    if (uploadFormVehicle) {
        uploadFormVehicle.addEventListener("submit", async (e) => {
            e.preventDefault();

            // 👉 STOP MODE
            if (isVehicleRunning) {
                await fetch('/stop_video', { method: 'POST' }).catch(() => { });

                if (videoFeedVehicle) {
                    videoFeedVehicle.src = "";
                    videoFeedVehicle.style.display = "none";
                }
                if (vehiclePlaceholder) vehiclePlaceholder.style.display = 'block';

                setVehicleStatus('Ready', '');
                if (uploadBtnText) uploadBtnText.textContent = 'Detect Vehicles';
                isVehicleRunning = false;

                // Keep button enabled if file still selected
                if (videoInput && videoInput.files.length > 0) {
                    uploadBtnVehicle.disabled = false;
                    uploadBtnVehicle.style.opacity = '1';
                    uploadBtnVehicle.style.cursor = 'pointer';
                }
                return;
            }

            // 👉 START MODE
            let file = videoInput.files[0];

            if (!file) {
                alert("Please select a video");
                return;
            }

            if (file.size > 20 * 1024 * 1024) {
                alert("File must be less than 20MB");
                return;
            }

            uploadBtnVehicle.disabled = true;
            uploadBtnVehicle.style.opacity = '0.5';
            uploadBtnVehicle.style.cursor = 'not-allowed';
            if (uploadBtnText) uploadBtnText.textContent = 'Uploading...';
            if (uploadSpinner) uploadSpinner.style.display = 'block';

            const formData = new FormData();
            formData.append("video", file);

            try {
                const response = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.filename) {
                    setVehicleStatus('Detecting vehicles...', 'success');
                    if (uploadBtnText) uploadBtnText.textContent = 'Stop Detection';
                    uploadBtnVehicle.disabled = false;
                    uploadBtnVehicle.style.opacity = '1';
                    uploadBtnVehicle.style.cursor = 'pointer';
                    isVehicleRunning = true;

                    if (vehiclePlaceholder) vehiclePlaceholder.style.display = 'none';

                    if (videoFeedVehicle) {
                        videoFeedVehicle.style.display = 'block';
                        videoFeedVehicle.src = "/video_feed/" + data.filename;
                    }

                } else {
                    alert("Upload failed");
                    setVehicleStatus(data.error || 'Upload failed', 'error');
                    uploadBtnVehicle.disabled = false;
                    uploadBtnVehicle.style.opacity = '1';
                    uploadBtnVehicle.style.cursor = 'pointer';
                    if (uploadBtnText) uploadBtnText.textContent = 'Detect Vehicles';
                }

            } catch (error) {
                console.error("Upload error:", error);
                alert("Upload error");
                setVehicleStatus('Error uploading video.', 'error');
                uploadBtnVehicle.disabled = false;
                uploadBtnVehicle.style.opacity = '1';
                uploadBtnVehicle.style.cursor = 'pointer';
                if (uploadBtnText) uploadBtnText.textContent = 'Detect Vehicles';
            } finally {
                if (uploadSpinner) uploadSpinner.style.display = 'none';
            }
        });
    }
});
