# SAM2 Mobile Server

A Flask-based REST API server that receives video frames and bounding boxes from mobile clients and returns segmentation results (bounding boxes and object IDs) using SAM2.

## Features

- **Initialize Sessions**: Start a new tracking session with an initial frame and bounding boxes
- **Track Frames**: Process subsequent video frames and get updated segmentation results
- **Binary Masks**: Optional binary segmentation masks for precise object annotation
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
  "object_ids": [1, 2, ...],  // Optional: custom object IDs
  "include_masks": true  // Optional: if true, returns binary masks in response
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], ...],
  "frame_shape": [height, width],
  "masks": ["base64_encoded_mask", ...]  // Optional: present only if include_masks=true
}
```

**Note:** The `masks` array contains base64-encoded binary masks (0s and 1s). Each mask corresponds to an object ID in the same order as `object_ids` and `bounding_boxes`. See the [Mask Decoding](#mask-decoding) section for details on how to decode masks.

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
  "image": "base64_encoded_image_or_data_url",
  "include_masks": true  // Optional: if true, returns binary masks in response
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], ...],  // null for lost objects
  "frame_index": 1,
  "frame_shape": [height, width],  // Frame dimensions (needed to decode masks)
  "masks": ["base64_encoded_mask", ...]  // Optional: present only if include_masks=true (null for lost objects)
}
```

**Note:** 
- Bounding boxes and masks are `null` for objects that are temporarily lost (not tracked in the current frame)
- The object IDs remain consistent even when objects are lost
- The `masks` array follows the same order as `object_ids` and `bounding_boxes`
- See the [Mask Decoding](#mask-decoding) section for details on how to decode masks

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

#### Basic Usage

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
    ],
    "include_masks": true  // Request binary masks
]

// Send POST request to http://your-server:8080/initialize
```

#### Response Models

```swift
import Foundation

struct BoundingBox: Codable {
    let x0: Double
    let y0: Double
    let x1: Double
    let y1: Double
    
    init(from decoder: Decoder) throws {
        var container = try decoder.singleValueContainer()
        let array = try container.decode([Double].self)
        guard array.count == 4 else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Bounding box must have 4 values")
        }
        x0 = array[0]
        y0 = array[1]
        x1 = array[2]
        y1 = array[3]
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode([x0, y0, x1, y1])
    }
}

struct InitializeResponse: Codable {
    let success: Bool
    let sessionId: String
    let objectIds: [Int]
    let boundingBoxes: [BoundingBox]
    let frameShape: [Int]
    let masks: [String?]?  // Optional: present only if include_masks=true
    
    enum CodingKeys: String, CodingKey {
        case success
        case sessionId = "session_id"
        case objectIds = "object_ids"
        case boundingBoxes = "bounding_boxes"
        case frameShape = "frame_shape"
        case masks
    }
}

struct TrackResponse: Codable {
    let success: Bool
    let sessionId: String
    let objectIds: [Int]
    let boundingBoxes: [BoundingBox?]  // Can be null for lost objects
    let frameIndex: Int
    let frameShape: [Int]
    let masks: [String?]?  // Optional: present only if include_masks=true (null for lost objects)
    
    enum CodingKeys: String, CodingKey {
        case success
        case sessionId = "session_id"
        case objectIds = "object_ids"
        case boundingBoxes = "bounding_boxes"
        case frameIndex = "frame_index"
        case frameShape = "frame_shape"
        case masks
    }
}
```

#### Mask Decoding

The masks are base64-encoded binary arrays where each byte is either `0` (background) or `1` (object). Here's how to decode them:

```swift
import Foundation

/// Represents a binary mask as a 2D array
typealias BinaryMask = [[UInt8]]

/// Decodes a base64-encoded mask string into a 2D binary mask array
/// - Parameters:
///   - encodedMask: Base64-encoded mask string from the API response
///   - height: Height of the mask (from frame_shape[0])
///   - width: Width of the mask (from frame_shape[1])
/// - Returns: 2D array where mask[row][col] is 0 (background) or 1 (object), or nil if decoding fails
func decodeMask(_ encodedMask: String, height: Int, width: Int) -> BinaryMask? {
    // Step 1: Decode base64 string to Data
    guard let maskData = Data(base64Encoded: encodedMask) else {
        print("Failed to decode base64 mask string")
        return nil
    }
    
    // Step 2: Convert Data to UInt8 array
    let byteArray = [UInt8](maskData)
    
    // Step 3: Validate size
    let expectedSize = height * width
    guard byteArray.count == expectedSize else {
        print("Mask size mismatch: expected \(expectedSize) bytes, got \(byteArray.count)")
        return nil
    }
    
    // Step 4: Reshape flat array into 2D array (row-major order)
    var mask: BinaryMask = []
    mask.reserveCapacity(height)
    
    for row in 0..<height {
        let startIndex = row * width
        let endIndex = startIndex + width
        let rowData = Array(byteArray[startIndex..<endIndex])
        mask.append(rowData)
    }
    
    return mask
}

/// Convenience function to decode mask from API response
func decodeMask(_ encodedMask: String?, frameShape: [Int]) -> BinaryMask? {
    guard let encodedMask = encodedMask,
          frameShape.count == 2 else {
        return nil
    }
    return decodeMask(encodedMask, height: frameShape[0], width: frameShape[1])
}
```

#### Complete Usage Example

```swift
// Example: Initialize session and decode masks
func initializeSession(image: UIImage, boundingBoxes: [[Double]], serverURL: String) async throws {
    // 1. Encode image
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw NSError(domain: "ImageEncoding", code: -1)
    }
    let base64Image = imageData.base64EncodedString()
    
    // 2. Prepare request
    let requestBody: [String: Any] = [
        "session_id": "session_\(UUID().uuidString)",
        "image": "data:image/jpeg;base64,\(base64Image)",
        "bounding_boxes": boundingBoxes,
        "include_masks": true
    ]
    
    // 3. Send request (using URLSession)
    let url = URL(string: "\(serverURL)/initialize")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    
    // 4. Decode response
    let response = try JSONDecoder().decode(InitializeResponse.self, from: data)
    
    // 5. Decode masks
    if let masks = response.masks {
        for (index, maskString) in masks.enumerated() {
            let objectId = response.objectIds[index]
            
            if let maskString = maskString {
                // Object is tracked - decode mask
                if let mask = decodeMask(maskString, frameShape: response.frameShape) {
                    print("Object ID \(objectId): Mask decoded successfully")
                    print("  Mask dimensions: \(mask.count) x \(mask[0].count)")
                    print("  Mask size: \(mask.count * mask[0].count) pixels")
                    
                    // Use mask for annotation/visualization
                    // mask[row][col] is 0 (background) or 1 (object)
                } else {
                    print("Object ID \(objectId): Failed to decode mask")
                }
            } else {
                // Object is lost (shouldn't happen in initialize, but handle gracefully)
                print("Object ID \(objectId): No mask available")
            }
        }
    }
}

// Example: Track frame and handle lost objects
func trackFrame(sessionId: String, image: UIImage, serverURL: String) async throws {
    // 1. Encode image
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw NSError(domain: "ImageEncoding", code: -1)
    }
    let base64Image = imageData.base64EncodedString()
    
    // 2. Prepare request
    let requestBody: [String: Any] = [
        "session_id": sessionId,
        "image": "data:image/jpeg;base64,\(base64Image)",
        "include_masks": true
    ]
    
    // 3. Send request
    let url = URL(string: "\(serverURL)/track")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    
    // 4. Decode response
    let response = try JSONDecoder().decode(TrackResponse.self, from: data)
    
    // 5. Process results
    if let masks = response.masks {
        for (index, maskString) in masks.enumerated() {
            let objectId = response.objectIds[index]
            let bbox = response.boundingBoxes[index]
            
            if let maskString = maskString, let bbox = bbox {
                // Object is tracked - decode mask
                if let mask = decodeMask(maskString, frameShape: response.frameShape) {
                    print("Frame \(response.frameIndex): Object ID \(objectId) tracked")
                    // Use mask and bbox for annotation
                }
            } else {
                // Object is lost
                print("Frame \(response.frameIndex): Object ID \(objectId) lost")
                // Handle lost object (e.g., hide annotation, show warning)
            }
        }
    }
}
```

#### Using Masks for Visualization

Once decoded, you can use the masks to create overlays on your images:

```swift
import UIKit

/// Creates a colored overlay image from a binary mask
func createMaskOverlay(mask: BinaryMask, color: UIColor, alpha: CGFloat) -> UIImage? {
    let height = mask.count
    let width = mask[0].count
    
    // Create a bitmap context
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bitsPerComponent = 8
    
    var pixelData = [UInt8](repeating: 0, count: height * width * bytesPerPixel)
    
    // Extract color components
    var red: CGFloat = 0
    var green: CGFloat = 0
    var blue: CGFloat = 0
    var alphaComponent: CGFloat = 0
    color.getRed(&red, green: &green, blue: &blue, alpha: &alphaComponent)
    
    // Fill pixels where mask is 1
    for row in 0..<height {
        for col in 0..<width {
            if mask[row][col] == 1 {
                let pixelIndex = (row * width + col) * bytesPerPixel
                pixelData[pixelIndex] = UInt8(red * 255)      // R
                pixelData[pixelIndex + 1] = UInt8(green * 255) // G
                pixelData[pixelIndex + 2] = UInt8(blue * 255)  // B
                pixelData[pixelIndex + 3] = UInt8(alpha * 255) // A
            }
        }
    }
    
    guard let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return nil
    }
    
    guard let cgImage = context.makeImage() else {
        return nil
    }
    
    return UIImage(cgImage: cgImage)
}
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

## Mask Decoding

### Mask Format

Masks are returned as base64-encoded binary arrays when `include_masks: true` is set in the request. Each mask is a flat array of bytes where:
- `0` = background pixel
- `1` = object pixel

The masks are stored in **row-major order** (first `width` bytes are row 0, next `width` bytes are row 1, etc.).

### Decoding Process

To decode a mask, you need:
1. The base64-encoded mask string from the API response
2. The frame dimensions from `frame_shape` (format: `[height, width]`)

**Decoding steps:**
1. Decode the base64 string to bytes
2. Convert bytes to a 1D array of `UInt8` values (0 or 1)
3. Validate the array size: `array.length == height * width`
4. Reshape the 1D array into a 2D array using row-major order:
   - Row `i` starts at index `i * width`
   - Row `i` contains `width` consecutive bytes

### Example Decoding (Pseudo-code)

```
Input:
  encodedMask = "AQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyAhIiMkJSYnKCkqKywtLi8wMTIzNDU2Nzg5Ojs8PT4/QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1+f4CBgoOEhYaHiImKi4yNjo+QkZKTlJWWl5iZmpucnZ6foKGio6SlpqeoqaqrrK2ur7CxsrO0tba3uLm6u7y9vr/AwcLDxMXGx8jJysvMzc7P0NHS09TV1tfY2drb3N3e3+Dh4uPk5ebn6Onq6+zt7u/w8fLz9PX29/j5+vv8/f7/"
  frameShape = [480, 640]  // height=480, width=640

Steps:
  1. Decode base64 → bytes (307,200 bytes for 480×640)
  2. Convert to UInt8 array → [0, 1, 0, 1, ...]
  3. Validate: 307,200 == 480 * 640 ✓
  4. Reshape:
     - Row 0: bytes[0..639]
     - Row 1: bytes[640..1279]
     - Row 2: bytes[1280..1919]
     - ...
     - Row 479: bytes[306,560..307,199]
```

### Important Notes

- **Mask Order**: Masks in the `masks` array correspond to `object_ids` in the same order
- **Lost Objects**: When an object is lost, its mask is `null` (same as its bounding box)
- **Consistent IDs**: Object IDs remain consistent even when objects are temporarily lost
- **Frame Shape**: Always use `frame_shape` from the response to decode masks, as frame dimensions may vary

## Notes

- **Image Format**: Images can be sent as base64-encoded strings or data URLs (e.g., `data:image/jpeg;base64,...`)
- **Bounding Box Format**: Bounding boxes are in format `[x0, y0, x1, y1]` where `(x0, y0)` is top-left and `(x1, y1)` is bottom-right
- **Object IDs**: If not provided, object IDs are auto-generated starting from 1. IDs remain consistent throughout the session, even if objects are temporarily lost
- **Masks**: Binary masks are optional and must be explicitly requested with `include_masks: true`. See [Mask Decoding](#mask-decoding) for details
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

