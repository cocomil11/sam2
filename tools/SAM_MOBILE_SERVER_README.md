# SAM2 Mobile Server

A Flask-based REST API server that receives video frames and bounding boxes from mobile clients and returns segmentation results (bounding boxes and object IDs) using SAM2.

## Features

- **Initialize Sessions**: Start a new tracking session with an initial frame and bounding boxes
- **Track Frames**: Process subsequent video frames and get updated segmentation results
- **Session Management**: Support multiple concurrent sessions
- **CORS Enabled**: Ready for mobile client integration

## Installation

Make sure you have the required dependencies:

```bash
pip install flask flask-cors opencv-python numpy torch
```

## Usage

### Starting the Server

**Important:** The server must run **inside WSL** (where SAM2 is installed).

```bash
# Inside WSL
python tools/sam_mobile_server.py \
    --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
    --checkpoint checkpoints/sam2.1_hiera_small.pt \
    --port 8080 \
    --device cuda
```

The server binds to `0.0.0.0` by default, making it accessible from both WSL and Windows.

### API Endpoints

#### 1. Health Check

**GET** `/health`

Check if the server is running and the predictor is loaded.

**Response:**
```json
{
  "status": "ok",
  "predictor_loaded": true
}
```

#### 2. Initialize Session

**POST** `/initialize`

Initialize a new tracking session with an initial frame and bounding boxes.

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "image": "base64_encoded_image_or_data_url",
  "bounding_boxes": [[x0, y0, x1, y1], ...],
  "object_ids": [1, 2, ...]  // Optional: custom object IDs
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], ...],
  "frame_shape": [height, width]
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my_session",
    "image": "data:image/jpeg;base64,/9j/4AAQ...",
    "bounding_boxes": [[100, 100, 200, 200], [300, 300, 400, 400]]
  }'
```

#### 3. Track Frame

**POST** `/track`

Track objects in a new frame from the video stream.

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "image": "base64_encoded_image_or_data_url"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], ...],
  "frame_index": 1
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/track \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my_session",
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

#### 4. Reset Session

**POST** `/reset`

Reset a tracking session (clears the predictor state).

**Request Body:**
```json
{
  "session_id": "unique_session_id"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Session my_session reset"
}
```

#### 5. List Sessions

**GET** `/sessions`

List all active sessions.

**Response:**
```json
{
  "sessions": ["session1", "session2", ...],
  "count": 2
}
```

## Mobile Client Integration

### Android (Kotlin Example)

```kotlin
// Initialize session
val imageBytes = // Get image as ByteArray
val base64Image = Base64.encodeToString(imageBytes, Base64.NO_WRAP)

val request = JSONObject().apply {
    put("session_id", "my_session_${System.currentTimeMillis()}")
    put("image", "data:image/jpeg;base64,$base64Image")
    put("bounding_boxes", JSONArray().apply {
        put(JSONArray(listOf(100, 100, 200, 200)))
        put(JSONArray(listOf(300, 300, 400, 400)))
    })
}

// Send POST request to http://your-server:8080/initialize
```

### iOS (Swift Example)

```swift
// Initialize session
let imageData = // Get image as Data
let base64Image = imageData.base64EncodedString()

let request: [String: Any] = [
    "session_id": "my_session_\(Date().timeIntervalSince1970)",
    "image": "data:image/jpeg;base64,\(base64Image)",
    "bounding_boxes": [
        [100, 100, 200, 200],
        [300, 300, 400, 400]
    ]
]

// Send POST request to http://your-server:8080/initialize
```

## Testing

### Option 1: Test from WSL (Easiest)

```bash
# Inside WSL
python tools/test_mobile_server.py \
    --server-url http://localhost:8080 \
    --image path/to/test_image.jpg \
    --bbox 100 100 200 200 \
    --bbox 300 300 400 400
```

### Option 2: Test from Windows

If running the test client from Windows, you need to use the WSL IP address instead of `localhost`:

1. **Get the WSL IP address:**
   ```bash
   # Inside WSL
   ./tools/get_wsl_ip.sh
   # Or manually:
   ip route show default | grep -oP 'via \K\S+'
   ```

2. **Run the test client from Windows:**
   ```bash
   # On Windows (PowerShell or CMD)
   python tools/test_mobile_server.py \
       --server-url http://<WSL_IP>:8080 \
       --image path/to/test_image.jpg \
       --bbox 100 100 200 200
   ```

   Replace `<WSL_IP>` with the IP address from step 1 (e.g., `172.21.128.1`).

## Notes

- **Image Format**: Images can be sent as base64-encoded strings or data URLs (e.g., `data:image/jpeg;base64,...`)
- **Bounding Box Format**: Bounding boxes are in format `[x0, y0, x1, y1]` where `(x0, y0)` is top-left and `(x1, y1)` is bottom-right
- **Object IDs**: If not provided, object IDs are auto-generated starting from 1
- **Session Management**: Each session maintains its own tracking state. Make sure to use the same `session_id` for all frames in a video sequence
- **Thread Safety**: The server uses locks to ensure thread-safe access to the predictor

## Error Handling

The server returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing parameters, invalid data)
- `404`: Session not found
- `500`: Server error (predictor not initialized, processing error)

Error responses include an `error` field with a description:
```json
{
  "error": "Session my_session not found. Please initialize first."
}
```

