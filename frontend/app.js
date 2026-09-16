const WS_URL = "ws://127.0.0.1:8000/ws/eeg-stream";

// DOM Elements
const connStatus = document.getElementById("connection-status");
const alertBanner = document.getElementById("alert-banner");
const probVal = document.getElementById("prob-val");
const epochsCount = document.getElementById("epochs-count");
const truthLabel = document.getElementById("truth-label");
const epochIndex = document.getElementById("epoch-index");

// Web Audio API for Alert Beep
let audioCtx = null;
function playAlertBeep() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880Hz pitch
    gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.5); // 500ms beep
}

// Chart.js Setup for 4 Channels
const ctx = document.getElementById("eegChart").getContext("2d");
const timePoints = Array.from({ length: 384 }, (_, i) => (i / 128).toFixed(2) + "s");

const eegChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: timePoints,
        datasets: [
            { label: "Fz", data: [], borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
            { label: "Cz", data: [], borderColor: "#a855f7", borderWidth: 1.5, pointRadius: 0 },
            { label: "Pz", data: [], borderColor: "#f59e0b", borderWidth: 1.5, pointRadius: 0 },
            { label: "Oz", data: [], borderColor: "#10b981", borderWidth: 1.5, pointRadius: 0 }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Turn off animation for snappy 3s epoch updates
        scales: {
            x: {
                display: true,
                title: { display: true, text: "Time (seconds)", color: "#94a3b8" },
                ticks: { color: "#94a3b8", maxTicksLimit: 10 },
                grid: { color: "#334155" }
            },
            y: {
                display: true,
                title: { display: true, text: "Amplitude (µV)", color: "#94a3b8" },
                ticks: { color: "#94a3b8" },
                grid: { color: "#334155" }
            }
        },
        plugins: {
            legend: { labels: { color: "#f8fafc" } }
        }
    }
});

// WebSocket Connection
const socket = new WebSocket(WS_URL);

socket.onopen = () => {
    connStatus.textContent = "LIVE STREAM ACTIVE";
    connStatus.style.backgroundColor = "#22c55e";
};

socket.onclose = () => {
    connStatus.textContent = "DISCONNECTED";
    connStatus.style.backgroundColor = "#ef4444";
};

socket.onerror = (err) => {
    console.error("WebSocket Error:", err);
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // 1. Update Metrics Cards
    const probPercentage = (data.drowsy_probability * 100).toFixed(1);
    probVal.textContent = `${probPercentage}%`;
    probVal.style.color = data.drowsy_probability >= 0.75 ? "#ef4444" : "#38bdf8";

    epochsCount.textContent = `${data.consecutive_drowsy_epochs} / 3`;
    truthLabel.textContent = data.ground_truth === 1 ? "DROWSY" : "ALERT";
    truthLabel.style.color = data.ground_truth === 1 ? "#ef4444" : "#22c55e";
    epochIndex.textContent = `#${data.epoch_index}`;

    // 2. Alert Engine UI & Sound Trigger
    if (data.alert) {
        alertBanner.style.display = "block";
        playAlertBeep();
    } else {
        alertBanner.style.display = "none";
    }

    // 3. Update Chart Datasets
    eegChart.data.datasets[0].data = data.channels.Fz;
    eegChart.data.datasets[1].data = data.channels.Cz;
    eegChart.data.datasets[2].data = data.channels.Pz;
    eegChart.data.datasets[3].data = data.channels.Oz;
    eegChart.update();
};