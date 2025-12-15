// ======================================================
// SINGLE MASK EDITOR - versione semplificata (una sola maschera)
// ======================================================

// ---------- DOM ELEMENTS ----------
const canvasElement = document.getElementById("canvas");
const bgCanvasElement = document.getElementById("bgCanvas");

const fileInput = document.getElementById("upload");
const undoButton = document.getElementById("undo");
const redoButton = document.getElementById("redo");
const removeBgButton = document.getElementById("removeBackground");
const clearButton = document.getElementById("clear");
const downloadButton = document.getElementById("downloadBtn");
const resultsSection = document.getElementById("resultsSection");
const resultsGrid = document.getElementById("resultsGrid");

let hasBackgroundImage = false;

// --- form e campi hidden per la POST ---
const generationForm = document.getElementById("generationForm");
const startButtons = document.querySelectorAll(".js-startGeneration");

const promptTextarea = document.getElementById("textPrompt");
const numGenerationsInput = document.getElementById("numGenerations");

const hiddenPrompt = document.getElementById("hiddenPrompt");
const hiddenNumGenerations = document.getElementById("hiddenNumGenerations");
const hiddenOriginalImage = document.getElementById("hiddenOriginalImage");
const hiddenMaskImage = document.getElementById("hiddenMaskImage");

// ---------- STATUS DISPLAY ----------
const statusDiv = document.getElementById("generation-status");
const statusText = document.getElementById("status-text");

// Se qualcosa di fondamentale manca, fermati.
if (!canvasElement || !bgCanvasElement) {
    console.error("Canvas element(s) not found. Controlla id='canvas' e id='bgCanvas'.");
}

// ---------- CANVAS & CONTEXT ----------
const context = canvasElement.getContext("2d");
const bgCtx = bgCanvasElement.getContext("2d");

// Assicuriamoci che i due canvas abbiano le stesse dimensioni logiche
bgCanvasElement.width = canvasElement.width;
bgCanvasElement.height = canvasElement.height;

// ---------- STATO DI DISEGNO ----------
let isDrawing = false;
let brushSize = 25;
let brushColor = "black";   // colore di default per la maschera
let tool = "pen";     // "pen" o "eraser"

// slider dimensione
const sizeElement = document.getElementById("sizeRange");
if (sizeElement) {
    brushSize = Number(sizeElement.value) || 25;
    sizeElement.addEventListener("input", (e) => {
        brushSize = Number(e.target.value) || brushSize;
    });
}

// radio per tool (pen / eraser)
const toolRadios = document.getElementsByName("toolRadio");
if (toolRadios && toolRadios.length > 0) {
    toolRadios.forEach((t) => {
        if (t.checked) {
            tool = t.value;
        }
        t.addEventListener("click", () => {
            tool = t.value;
        });
    });
}

// radio per colore (opzionale: se non esistono, resta "black")
const colorRadios = document.getElementsByName("colorRadio");
if (colorRadios && colorRadios.length > 0) {
    colorRadios.forEach((c) => {
        if (c.checked) {
            brushColor = c.value;
        }
        c.addEventListener("click", () => {
            brushColor = c.value;
        });
    });
}

// ---------- UNDO / REDO ----------
const undoStack = [];
const redoStack = [];
const maxHistory = 50;

// salva stato corrente del canvas principale
function saveState(clearRedo = true) {
    try {
        const snapshot = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
        undoStack.push(snapshot);
        if (undoStack.length > maxHistory) {
            undoStack.shift();
        }
        if (clearRedo) {
            redoStack.length = 0;
        }
    } catch (e) {
        console.warn("Impossibile salvare lo stato del canvas:", e);
    }
}

// ripristina uno stato (ImageData)
function restoreState(imageData, ctx) {
    if (!imageData || !ctx) return;
    ctx.putImageData(imageData, 0, 0);
}

// bottoni undo/redo
if (undoButton) {
    undoButton.addEventListener("click", () => {
        if (undoStack.length === 0) return;
        const last = undoStack.pop();
        try {
            const current = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
            redoStack.push(current);
        } catch (e) {
        }
        restoreState(last, context);
    });
}

if (redoButton) {
    redoButton.addEventListener("click", () => {
        if (redoStack.length === 0) return;
        const next = redoStack.pop();
        try {
            const current = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
            undoStack.push(current);
        } catch (e) {
        }
        restoreState(next, context);
    });
}

// scorciatoie da tastiera: Ctrl/Cmd+Z + Ctrl/Cmd+Y / Shift+Z
window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
            // Ctrl+Shift+Z -> redo
            if (redoButton) redoButton.click();
        } else {
            // Ctrl+Z -> undo
            if (undoButton) undoButton.click();
        }
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault();
        if (redoButton) redoButton.click();
    }
});

// ---------- CLEAR ----------
if (clearButton) {
    clearButton.addEventListener("click", () => {
        saveState(true);
        context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    });
}

// ---------- FUNZIONI UTILI ----------

// coordinate mouse relative al canvas
function getPos(e) {
    const rect = canvasElement.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

// disegno dell'immagine di sfondo su bgCanvas con "cover + crop centrale"
function drawImageCover(ctx, img, cw, ch) {
    if (!img) return;

    if (typeof ctx.imageSmoothingEnabled !== "undefined") {
        ctx.imageSmoothingEnabled = true;
    }
    try {
        ctx.imageSmoothingQuality = "high";
    } catch (e) {
    }

    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;

    if (!iw || !ih) {
        ctx.drawImage(img, 0, 0, cw, ch);
        return;
    }

    const canvasRatio = cw / ch;
    const imgRatio = iw / ih;

    let sx, sy, sWidth, sHeight;

    if (imgRatio > canvasRatio) {
        // immagine più larga
        sHeight = ih;
        sWidth = Math.round(ih * canvasRatio);
        sx = Math.round((iw - sWidth) / 2);
        sy = 0;
    } else {
        // immagine più alta o uguale
        sWidth = iw;
        sHeight = Math.round(iw / canvasRatio);
        sx = 0;
        sy = Math.round((ih - sHeight) / 2);
    }

    ctx.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, cw, ch);
}

// trasformazione file→dataURL
function fileToDataUri(field) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            resolve(reader.result);
        });
        reader.readAsDataURL(field);
    });
}

// ---------- UPLOAD IMMAGINE ----------
if (fileInput) {
    fileInput.addEventListener("change", async () => {
        const [file] = fileInput.files;
        if (!file) return;

        const img = document.createElement("img");
        img.src = await fileToDataUri(file);

        img.addEventListener("load", () => {
            // disegna l'immagine di sfondo sul bgCanvas
            drawImageCover(bgCtx, img, bgCanvasElement.width, bgCanvasElement.height);
            hasBackgroundImage = true;
            // resetta la maschera e la history
            context.clearRect(0, 0, canvasElement.width, canvasElement.height);
            undoStack.length = 0;
            redoStack.length = 0;
        });
    });
}

// ---------- DRAWING SULLA MASCHERA ----------

canvasElement.addEventListener("mousedown", (e) => {
    saveState(true);

    isDrawing = true;
    context.beginPath();
    context.lineWidth = brushSize;
    context.lineCap = "round";
    context.lineJoin = "round";

    if (tool === "eraser") {
        // la gomma rende trasparenti i pixel esistenti
        context.globalCompositeOperation = "destination-out";
        context.strokeStyle = "rgba(0,0,0,1)"; // il colore non conta davvero qui
    } else {
        context.globalCompositeOperation = "source-over";
        context.strokeStyle = brushColor || "black";
    }

    const p = getPos(e);
    context.moveTo(p.x, p.y);
});

canvasElement.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    const p = getPos(e);
    context.lineTo(p.x, p.y);
    context.stroke();
});

canvasElement.addEventListener("mouseup", () => {
    if (!isDrawing) return;
    isDrawing = false;
    context.closePath();
});

canvasElement.addEventListener("mouseleave", () => {
    if (!isDrawing) return;
    isDrawing = false;
    context.closePath();
});

// ---------- DOWNLOAD ZIP (original + mask) ----------

// helper: aggiunge un ImageData come PNG (bianco/nero) allo zip
async function addImageDataBinaryToZip(zip, imageData, filename, invert = false) {
    const w = imageData.width;
    const h = imageData.height;

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = w;
    tmpCanvas.height = h;
    const tmpCtx = tmpCanvas.getContext("2d");

    const outImageData = tmpCtx.createImageData(w, h);
    const src = imageData.data;
    const dst = outImageData.data;

    // Uso l'alpha per distinguere "disegnato" (maschera) vs trasparente
    for (let i = 0; i < src.length; i += 4) {
        const a = src[i + 3]; // alpha 0..255
        let bw = a > 0 ? 0 : 255; // dentro mask=0 (nero), fuori=255 (bianco)
        if (invert) bw = 255 - bw;

        dst[i] = bw;
        dst[i + 1] = bw;
        dst[i + 2] = bw;
        dst[i + 3] = 255;
    }

    tmpCtx.putImageData(outImageData, 0, 0);
    const dataURL = tmpCanvas.toDataURL("image/png");
    const base64 = dataURL.split(",")[1];

    zip.file(filename, base64, {base64: true});
}

// helper: aggiunge un canvas allo zip come PNG
function addCanvasToZip(zip, canvas, filename) {
    const dataURL = canvas.toDataURL("image/png");
    const base64 = dataURL.split(",")[1];
    zip.file(filename, base64, {base64: true});
}

// click su "Download zip"
if (downloadButton) {
    downloadButton.addEventListener("click", async () => {
        if (!hasBackgroundImage) {
            alert("Carica prima un'immagine di sfondo.");
            return;
        }

        const zip = new JSZip();

        // 1) original.png -> background
        addCanvasToZip(zip, bgCanvasElement, "original.png");

        // 2) mask.png -> maschera corrente dal canvas principale
        const maskData = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
        await addImageDataBinaryToZip(zip, maskData, "mask.png", false);

        // genera lo zip e scarica
        const blob = await zip.generateAsync({type: "blob"});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "images.zip";
        a.click();

        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
}
// ======================================================
// Fine single-mask editor
// ======================================================


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith(name + "=")) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// ==================================================
// REMOVE BACKGROUND -> crea RemoveBgJob + polling
// ==================================================

let REMOVE_BG_JOB_ID = null;
let removeBgPollingInterval = null;
const REMOVE_BG_STATUS_BASE_URL = "/editor/remove-background-status/";

if (removeBgButton) {

    removeBgButton.addEventListener("click", async () => {
        if (!hasBackgroundImage) {
            alert("Carica prima un'immagine di sfondo.");
            return;
        }

        // 1) background canvas -> Blob PNG
        const blob = await new Promise(resolve =>
            bgCanvasElement.toBlob(resolve, "image/png")
        );
        const modelSelect = document.getElementById("rembgModel");
        const selectedModel = modelSelect.value
        // 2) preparo FormData
        const formData = new FormData();
        formData.append("image", blob, "original.png");
        formData.append("model", selectedModel);

        try {
            const response = await fetch("/editor/remove-background/", {
                method: "POST",
                headers: {
                    "X-CSRFToken": getCookie("csrftoken"),
                },
                body: formData,
            });

            const data = await response.json();

            // 3) controllo risposta
            if (!data.ok) {
                console.error("Errore server:", data.error);
                updateStatus("FAILED", data.error);
                return;
            }

            // 4) avvio polling
            updateStatus(data.status, data.error);
            startRemoveBgPolling(data.job_id);

        } catch (err) {
            console.error("Errore di rete:", err);
            updateStatus("FAILED", "Errore di rete durante remove background");
        }
    });
}

function drawImageOnWhiteBackground(ctx, img, cw, ch) {
    // 1) pulisco
    ctx.clearRect(0, 0, cw, ch);

    // 2) sfondo bianco
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, cw, ch);
    ctx.restore();

    // 3) immagine sopra (usa la tua funzione cover/crop)
    drawImageCover(ctx, img, cw, ch);
}

function startRemoveBgPolling(jobId) {
    REMOVE_BG_JOB_ID = jobId;

    if (removeBgPollingInterval) {
        clearInterval(removeBgPollingInterval);
    }

    // polling ogni 2 secondi
    removeBgPollingInterval = setInterval(checkRemoveBgStatus, 2000);
}


function checkRemoveBgStatus() {
    if (!REMOVE_BG_JOB_ID) return;

    fetch(`${REMOVE_BG_STATUS_BASE_URL}${REMOVE_BG_JOB_ID}/`)
        .then(r => r.json())
        .then(data => {
            updateStatus(data.status, data.error);

            if (data.status === "FAILED") {
                clearInterval(removeBgPollingInterval);
                return;
            }

            if (data.status === "COMPLETED") {
                clearInterval(removeBgPollingInterval);

                // Same-origin endpoint (niente CORS)
                const resultUrl = `/editor/remove-background-result/${REMOVE_BG_JOB_ID}/?t=${Date.now()}`;

                const img = new Image();

                img.onload = () => {
                    drawImageOnWhiteBackground(
                        bgCtx,
                        img,
                        bgCanvasElement.width,
                        bgCanvasElement.height
                    );

                    hasBackgroundImage = true;

                    // reset maschera e history
                    context.clearRect(0, 0, canvasElement.width, canvasElement.height);
                    undoStack.length = 0;
                    redoStack.length = 0;
                };

                img.onerror = (e) => {
                    console.error("Errore caricamento risultato remove-bg (same-origin):", resultUrl, e);
                };

                img.src = resultUrl;
            }
        })
        .catch(err => {
            console.error("Errore polling remove-bg:", err);
        });
}

// ==================================================
// GESTIONE STATUS GENERAZIONE
// ==================================================
function updateStatus(status, errorMessage = null) {
    // Mostra la box se non è visibile
    if (statusDiv.style.display === "none") {
        statusDiv.style.display = "block";
        statusText.textContent = " In attesa...";
    }

    // Reset classi
    statusDiv.classList.remove("failed", "completed", "running");

    // Applica la classe e il testo corretto
    if (status === "RUNNING") {
        statusDiv.classList.add("running");
        statusText.textContent = "Generazione in corso...";
    } else if (status === "COMPLETED") {
        statusDiv.classList.add("completed");
        statusText.textContent = "Generazione completata!";
    } else if (status === "FAILED") {
        statusDiv.classList.add("failed");
        statusText.textContent = `Errore nella generazione: ${errorMessage ?? ""}`;
    } else if (status === "PENDING") {
        statusText.textContent = "Job inviato, in attesa che Celery inizi...";
    }
}

// ==================================================
// START GENERATION -> prepara dati e invia la form
// ==================================================
const errorsBox = document.getElementById("formErrors");

if (startButtons.length && generationForm) {
    startButtons.forEach(btn => { // per ogni bottone "Start Generation"
        btn.addEventListener("click", () => { // click sul bottone
            /*
            Controlla che tutti gli input del form siano validi. Una volta verificato vengono
            presi i vari input e mandati a Django con un post
            * */
            statusDiv.style.display = "none";
            if (errorsBox) { // reset errori
                errorsBox.textContent = "";
            }
            // VALIDAZIONE
            const errors = [];
            // 1) serve un'immagine di sfondo
            if (!hasBackgroundImage) {
                errors.push("Carica un'immagine di sfondo prima di procedere.");
            }

            // 2) prompt e numero generazioni
            if (hiddenPrompt && promptTextarea) {
                if (!(promptTextarea?.value || "").trim()) {
                    errors.push("Inserisci una descrizione della modifica (prompt).");
                }
            }

            // 3) Controllo: numero generazioni valido
            const numGenValue = parseInt(numGenerationsInput?.value || "1", 10);
            if (isNaN(numGenValue) || numGenValue < 1 || numGenValue > 4) {
                errors.push("Il numero di varianti deve essere compreso tra 1 e 4.");
            }

            // Se ci sono errori -> MOSTRO e NON invio la form
            if (errors.length > 0) {
                if (errorsBox) {
                    const ul = document.createElement("ul");
                    errors.forEach(msg => {
                        const li = document.createElement("li");
                        li.textContent = msg;
                        ul.appendChild(li);
                    });
                    errorsBox.appendChild(ul);
                }
            } else {

                // Se tutto ok -> preparo hidden
                hiddenPrompt.value = (promptTextarea?.value || "").trim();
                hiddenNumGenerations.value = String(numGenValue);
                hiddenOriginalImage.value = bgCanvasElement.toDataURL("image/png");
                hiddenMaskImage.value = canvasElement.toDataURL("image/png");

                // ⚠️ QUI NON FACCIAMO PIÙ generationForm.submit()
                // Invece inviamo via fetch e restiamo sulla stessa pagina

                const formData = new FormData(generationForm);

                // opzionale: disabilito il bottone / mostro "caricamento"

                console.log("Form action:", generationForm.action);

                fetch(generationForm.action, {
                    method: generationForm.method || "POST",
                    body: formData
                })
                    .then(async response => {
                        console.log("HTTP status:", response.status);

                        const text = await response.text();
                        console.log("Raw response text:", text);

                        if (!response.ok) {
                            // Qui vedi il corpo dell'errore (es. pagina HTML con stacktrace)
                            throw new Error("Response not OK");
                        }

                        // Provo a parsare JSON manualmente
                        let data;
                        try {
                            data = JSON.parse(text);
                        } catch (e) {
                            console.error("Errore parse JSON:", e);
                            throw e;
                        }

                        if (!data.ok) {
                            if (errorsBox) {
                                errorsBox.textContent = data.error || "Errore nella richiesta.";
                            }
                            return;
                        }

                        const jobId = data.job_id;
                        console.log("JOB ID ricevuto:", jobId);
                        updateStatus(data.status, data.error);
                        startJobPolling(jobId);
                    })
                    .catch(err => {
                        console.error("Errore nella richiesta di generazione:", err);
                        if (errorsBox) {
                            errorsBox.textContent = "Errore di rete o server durante la generazione.";
                        }
                    });
            }
        });
    });
}
let CURRENT_JOB_ID = null;
let pollingInterval = null;

function startJobPolling(jobId) {
    CURRENT_JOB_ID = jobId; // memorizza l'ID del job corrente

    // Ogni 2 secondi controlliamo lo stato
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    // Avvia il polling periodico di 2 secondi, dove chiameremo checkJobStatus
    pollingInterval = setInterval(checkJobStatus, 2000);
}

const JOB_STATUS_BASE_URL = "/editor/job-status/";  // <-- prefisso fisso

function checkJobStatus() {
    if (!CURRENT_JOB_ID) return;
    //fetch chiama la view di django che ritorna lo stato del job
    fetch(`${JOB_STATUS_BASE_URL}${CURRENT_JOB_ID}/`)
        .then(r => r.json())
        //data contiene le informazioni sul job
        .then(data => {

            updateStatus(data.status, data.error)

            //se la generazione è FALLITA
            if (data.status === "FAILED") {
                clearInterval(pollingInterval);
                if (resultsSection) {
                    // opzionale: nascondi i vecchi risultati
                    resultsSection.style.display = "none";
                }
            }
            //se la generazione è COMPLETATA
            if (data.status === "COMPLETED") {
                clearInterval(pollingInterval);
                // Mostro la sezione risultati
                if (resultsSection) {
                    resultsSection.style.display = "block";
                }

                if (resultsGrid) {
                    // Svuoto la griglia
                    resultsGrid.innerHTML = "";

                    if (data.generated_images && data.generated_images.length > 0) {
                        data.generated_images.forEach((url, index) => {
                            const figure = document.createElement("figure");
                            figure.className = "result-item";

                            const wrapper = document.createElement("div");
                            wrapper.className = "result-image-wrapper";

                            const img = document.createElement("img");
                            img.src = url;
                            img.alt = `Variante ${index + 1}`;
                            img.loading = "lazy";

                            const caption = document.createElement("figcaption");
                            caption.className = "result-caption";
                            caption.textContent = `Variante ${index + 1}`;

                            wrapper.appendChild(img);
                            figure.appendChild(wrapper);
                            figure.appendChild(caption);
                            resultsGrid.appendChild(figure);
                        });

                        // Abilito il pulsante di download se ci sono immagini
                        if (downloadButton) {
                            downloadButton.disabled = false;
                        }
                    } else {
                        // Nessuna immagine trovata
                        const msg = document.createElement("p");
                        msg.textContent = "Nessuna immagine trovata.";
                        resultsGrid.appendChild(msg);

                        if (downloadButton) {
                            downloadButton.disabled = true;
                        }
                    }
                }
            }

        })
        .catch(err => {
            console.error("Errore durante il polling:", err);
        });
}
