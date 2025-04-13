import {
    PoseLandmarker,
    FilesetResolver,}
    from "@mediapipe/tasks-vision";

let poseLandmarker, nn;
let results = undefined;
let lastLandmarks = [];

const poseBtn = document.getElementById("webcamButton");
const statusText = document.getElementById('model-status');
const predictionText = document.getElementById('prediction-status');

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

async function init() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
            `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    delegate: "GPU"
},
    runningMode: "VIDEO",
        numPoses: 1
});


    ml5.setBackend("webgl");
    nn = ml5.neuralNetwork({ task: 'classification', debug: true });

    const modelDetails = {
        model: './models/model.json',
        metadata: './models/model_meta.json',
        weights: './models/model.weights.bin'
    };

    nn.load(modelDetails, () => {
        console.log("Neural network loaded");
        startWebcam();
    });
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;

            requestAnimationFrame(update)
        });
    } catch (err) {
        console.error("Fout bij toegang tot webcam:", err);
    }
}

async function update() {
    const results = await poseLandmarker.detectForVideo(video, performance.now());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks.length > 0) {
        const pose = results.landmarks[0];

        // Use only 6 values: Left shoulder and Right shoulder (3 values for each)
        const landmarks = [
            pose[11]?.x, pose[11]?.y, pose[11]?.z, // Left shoulder
            pose[12]?.x, pose[12]?.y, pose[12]?.z  // Right shoulder
        ];

        // Only update lastLandmarks if all coordinates are defined
        if (landmarks.every(coord => coord !== undefined)) {
            lastLandmarks = landmarks;
        }
    }

    requestAnimationFrame(update);
}




poseBtn.addEventListener("click", async () => {
    if (lastLandmarks.length === 6 && nn) {
        try {
            const results = await nn.classify(lastLandmarks);
            const best = results[0];
            predictionText.textContent =  `Recognized as: ${best.label} (${(best.confidence * 100).toFixed(1)}%) sure`;
            console.log(results);
        } catch (error) {
            predictionText.textContent = "Error loading";
            console.error("error loading classification:", error);
        }
    } else {
        predictionText.textContent = "There is no pose available.";
    }
});

if (navigator.mediaDevices?.getUserMedia) {
    init();
}
