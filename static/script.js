// ── Identity Engine — training helpers ───────────────────────────────────────
let images = [];
let activeFeature = null;  // 'face' | 'object' | null

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
    const typeElem = document.getElementById("person-type");
    if (!nameElem) return;
    const name = nameElem.value.trim();
    const personType = typeElem ? typeElem.value : "known";

    if (!name || images.length === 0) {
        setExternalStatus("Enter name and upload images", "error");
        setTimeout(() => setExternalStatus('Ready', ''), 2000);
        return;
    }

    setExternalStatus('Uploading and communicating with AI module...', 'info');

    const formData = new FormData();
    formData.append("name", name);
    formData.append("type", personType);
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


// ── Restricted Area — registration helpers ────────────────────────────────────
let raImages = [];

function raHandleUpload(files) {
    for (let f of files) raImages.push(f);
    raRenderPreviews();
}

function raRenderPreviews() {
    const container = document.getElementById('ra-preview');
    if (!container) return;
    container.innerHTML = '';
    raImages.forEach((file, i) => {
        const url = URL.createObjectURL(file);
        container.innerHTML += `
        <div class="img-wrapper">
            <img src="${url}" width="60">
            <button class="remove-btn" onclick="event.stopPropagation(); raRemoveImage(${i})">✖</button>
        </div>`;
    });
}

function raRemoveImage(index) {
    raImages.splice(index, 1);
    raRenderPreviews();
}

function setRaStatus(msg, type) {
    const el = document.getElementById('ra-register-status');
    if (el) { el.textContent = msg; el.className = 'status-box ' + type; }
}

function raRegisterPerson() {
    const name = (document.getElementById('ra-name') || {}).value?.trim();
    if (!name) { setRaStatus('Enter a name.', 'error'); return; }
    if (raImages.length === 0) { setRaStatus('Upload at least 1 image.', 'error'); return; }

    setRaStatus('Processing encodings...', 'info');
    const fd = new FormData();
    fd.append('name', name);
    raImages.forEach(img => fd.append('images', img));

    fetch('/add_known_person', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                setRaStatus(data.message, 'success');
                raImages = [];
                document.getElementById('ra-name').value = '';
                raRenderPreviews();
            } else {
                setRaStatus(data.message, 'error');
            }
            setTimeout(() => setRaStatus('Ready', ''), 5000);
        })
        .catch(() => { setRaStatus('Server error.', 'error'); });
}

function raRefreshAlerts() {
    fetch('/get_alerts')
        .then(r => r.json())
        .then(alerts => {
            const tbody = document.getElementById('alert-logs-body');
            if (!tbody) return;
            if (!alerts.length) {
                tbody.innerHTML = '<tr><td colspan="3" style="padding:18px;text-align:center;color:#555;">No alerts logged yet.</td></tr>';
                return;
            }
            tbody.innerHTML = alerts.map(a => `
                <tr>
                    <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);">${a.timestamp}</td>
                    <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:var(--danger);">&#x26A0; ${a.status}</td>
                    <td style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06);">
                        ${a.image_path ? `<a href="/${a.image_path}" target="_blank" style="color:var(--accent);">View</a>` : '—'}
                    </td>
                </tr>`).join('');
        })
        .catch(() => {});
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
        const zoneControls = document.getElementById('fr-zone-controls');
        const zoneCanvas   = document.getElementById('fr-zone-canvas');
        const zoneInstr    = document.getElementById('zone-instructions');
        if (zoneControls) zoneControls.style.display = 'none';
        if (zoneCanvas) {
            zoneCanvas.style.display = 'none';
            const ctx = zoneCanvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
        }
        if (zoneInstr) zoneInstr.style.display = 'none';
        setStatus('Ready');
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

    // ── Restricted Area Security controls ─────────────────────────────────────
    const restrictedStartBtn = document.getElementById('restricted-start-btn');
    const restrictedStopBtn = document.getElementById('restricted-stop-btn');
    const restrictedFeed = document.getElementById('restricted-feed');
    const restrictedPlaceholder = document.getElementById('restricted-placeholder');
    const restrictedIndicator = document.getElementById('restricted-indicator');
    const restrictedStatus = document.getElementById('restricted-status');

    function setRestrictedStatus(msg, type) {
        if (!restrictedStatus) return;
        restrictedStatus.textContent = msg;
        restrictedStatus.className = 'status-box ' + type;
    }

    function resetRestrictedUI() {
        if (restrictedFeed) { restrictedFeed.src = ''; restrictedFeed.style.display = 'none'; }
        if (restrictedPlaceholder) restrictedPlaceholder.style.display = 'flex';
        if (restrictedIndicator) { restrictedIndicator.classList.remove('active'); restrictedIndicator.style.color = 'var(--text-dim)'; restrictedIndicator.textContent = 'READY'; }
        if (restrictedStartBtn) restrictedStartBtn.style.display = 'flex';
        if (restrictedStopBtn) restrictedStopBtn.style.display = 'none';
        setRestrictedStatus('Security Offline');
    }

    // ── Feature isolation — only one stream at a time ─────────────────────────
    async function stopAllFeatures() {
        if (activeFeature === 'face') {
            await fetch('/stop_fr_camera', { method: 'POST' }).catch(() => { });
            resetFaceUI();
        } else if (activeFeature === 'object') {
            await fetch('/stop_object_camera', { method: 'POST' }).catch(() => { });
            resetObjectUI();
        } else if (activeFeature === 'restricted') {
            await fetch('/stop_restricted_camera', { method: 'POST' }).catch(() => { });
            resetRestrictedUI();
        }
        activeFeature = null;
    }

    // ── Face Recognition — event listeners ───────────────────────────────────
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            const sourceType = document.getElementById('fr-source-type')?.value || 'webcam';

            await stopAllFeatures();
            setStatus('Starting camera...', 'info');
            try {
                const resp = await fetch('/start_fr_camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: sourceType })
                });
                const result = await resp.json();
                if (result.success) {
                    activeFeature = 'face';
                    if (placeholder) placeholder.style.display = 'none';
                    if (videoFeed) { videoFeed.style.display = 'block'; videoFeed.src = `/fr_video_feed?t=${Date.now()}`; }
                    if (indicator) { indicator.classList.add('active'); indicator.style.color = 'var(--success)'; indicator.textContent = 'DETECTING...'; }
                    if (startBtn) startBtn.style.display = 'none';
                    if (stopBtn) stopBtn.style.display = 'flex';
                    
                    const zoneControls = document.getElementById('fr-zone-controls');
                    const zoneCanvas   = document.getElementById('fr-zone-canvas');
                    const zoneInstr    = document.getElementById('zone-instructions');
                    if (zoneControls) zoneControls.style.display = 'flex';
                    if (zoneInstr)    zoneInstr.style.display = 'block';
                    if (zoneCanvas) {
                        zoneCanvas.style.display = 'block';
                        // Let layout reflow before syncing canvas resolution
                        setTimeout(() => {
                            if (typeof window.startDrawingZone === 'function') {
                                window.startDrawingZone('fr');
                            }
                        }, 120);
                    }

                    setStatus('Camera Active. Click video to draw polygon zone.', 'success');
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
                await fetch('/stop_fr_camera', { method: 'POST' });
                activeFeature = null;
                resetFaceUI();
                setStatus('Camera stopped.', 'info');
                setTimeout(() => setStatus('Ready', ''), 3000);
            } catch (err) {
                setStatus('Error stopping camera.', 'error');
            }
        });
    }


    // ── Object Detection — event listeners ────────────────────────────────────
    if (objectStartBtn) {
        objectStartBtn.addEventListener('click', async () => {
            const sourceType = document.getElementById('object-source-type')?.value || 'webcam';

            await stopAllFeatures();
            setObjectStatus('Starting object detection...', 'info');
            try {
                const resp = await fetch('/start_object_camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: sourceType })
                });
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

    // ── Restricted Area Security — event listeners ────────────────────────────
    if (restrictedStartBtn) {
        restrictedStartBtn.addEventListener('click', async () => {
            const sourceType = document.getElementById('restricted-source-type')?.value || 'webcam';

            await stopAllFeatures();
            setRestrictedStatus('Starting security feed...', 'info');
            try {
                const resp = await fetch('/start_restricted_camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: sourceType })
                });
                const result = await resp.json();
                if (result.success) {
                    activeFeature = 'restricted';
                    if (restrictedPlaceholder) restrictedPlaceholder.style.display = 'none';
                    if (restrictedFeed) { restrictedFeed.style.display = 'block'; restrictedFeed.src = `/restricted_video_feed?t=${Date.now()}`; }
                    if (restrictedIndicator) { restrictedIndicator.classList.add('active'); restrictedIndicator.style.color = 'var(--danger)'; restrictedIndicator.textContent = 'MONITORING...'; }
                    if (restrictedStartBtn) restrictedStartBtn.style.display = 'none';
                    if (restrictedStopBtn) restrictedStopBtn.style.display = 'flex';
                    setRestrictedStatus('Security Monitoring Active.', 'success');
                } else {
                    setRestrictedStatus(result.message || 'Could not start camera.', 'error');
                }
            } catch (err) {
                setRestrictedStatus('Could not start security camera.', 'error');
            }
        });
    }

    if (restrictedStopBtn) {
        restrictedStopBtn.addEventListener('click', async () => {
            try {
                await fetch('/stop_restricted_camera', { method: 'POST' });
                activeFeature = null;
                resetRestrictedUI();
                setRestrictedStatus('Security monitoring offline.', 'info');
                setTimeout(() => setRestrictedStatus('Ready', ''), 3000);
            } catch (err) {
                setRestrictedStatus('Error stopping camera.', 'error');
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
                        videoFeedVehicle.src = "/vehicle_video_feed/" + data.filename;
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

    // ── Camera Source UI Toggle ──────────────────────────────────────────────
    window.toggleCardCameraInput = function(prefix) {
        const typeSelect = document.getElementById(`${prefix}-source-type`);
        const cctvContainer = document.getElementById(`${prefix}-cctv-container`);
        if (typeSelect && cctvContainer) {
            cctvContainer.style.display = typeSelect.value === 'cctv' ? 'block' : 'none';
        }
    };

    // ── Zone Drawing Stubs ───────────────────────────────────────────────────
    // Polygon zone drawing is handled by the page-level script in face_recognition.html.
    // These stubs prevent errors on pages that don't include the polygon engine.
    if (typeof window.startDrawingZone === 'undefined') {
        window.startDrawingZone = function(prefix) {};
    }
    if (typeof window.saveZone === 'undefined') {
        window.saveZone = async function(prefix) {};
    }
    if (typeof window.clearZone === 'undefined') {
        window.clearZone = async function(prefix) {};
    }
    if (typeof window.undoLastPoint === 'undefined') {
        window.undoLastPoint = function(prefix) {};
    }

});
