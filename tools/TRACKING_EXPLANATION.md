# How SAM2 Tracking Works in the Mobile Server

## Overview

The SAM2 Mobile Server uses **temporal tracking** to follow objects across video frames. Here's how it works:

## Workflow

### 1. Initialization (`/initialize` endpoint)

**Mobile client sends:**
- First frame from video stream
- Bounding boxes defining what to track (e.g., `[[100, 100, 200, 200]]` for one object)

**Server does:**
1. Loads the first frame into SAM2's memory
2. For each bounding box:
   - Generates a segmentation mask for that region
   - Stores the object in SAM2's tracking memory
   - Assigns an object ID
3. Returns the initial segmentation results (bounding boxes extracted from masks)

**Key point:** Bounding boxes are **only needed in the first request**. They tell SAM2 "track these objects."

### 2. Tracking (`/track` endpoint)

**Mobile client sends:**
- New frame from video stream
- **No bounding boxes needed!**

**Server does:**
1. Uses SAM2's internal memory (from previous frames)
2. Automatically tracks the objects defined during initialization
3. Returns updated bounding boxes and object IDs

**Key point:** SAM2 maintains **temporal memory** across frames, so it can:
- Track objects as they move
- Handle appearance changes
- Recover from temporary occlusions
- Maintain object identity across frames

## How SAM2 Tracks Objects

SAM2 uses a **memory-based tracking** approach:

1. **Initial Frame (Frame 0):**
   - Bounding boxes define objects to track
   - SAM2 generates segmentation masks
   - Stores visual features and masks in memory

2. **Subsequent Frames (Frame 1, 2, 3...):**
   - SAM2 compares new frame with memory
   - Uses attention mechanisms to find the same objects
   - Updates masks and bounding boxes
   - Maintains object IDs consistently

3. **Memory Management:**
   - SAM2 keeps a sliding window of recent frames
   - Uses these frames to maintain tracking accuracy
   - Automatically handles object movement and appearance changes

## Example Flow

```
Frame 0 (Initialize):
  Client → Server: Frame + BBox [100, 100, 200, 200]
  Server → Client: Object ID: 1, BBox: [98, 102, 202, 198]  (refined from mask)

Frame 1 (Track):
  Client → Server: Frame (no bbox needed!)
  Server → Client: Object ID: 1, BBox: [105, 108, 207, 204]  (object moved)

Frame 2 (Track):
  Client → Server: Frame (no bbox needed!)
  Server → Client: Object ID: 1, BBox: [110, 115, 212, 209]  (object moved more)

... and so on
```

## Important Notes

### Single Session Limitation

Currently, the server uses a **single global predictor instance**. This means:
- Only **one active session** is supported at a time
- Initializing a new session resets the tracking state
- For multiple concurrent sessions, separate predictor instances would be needed

### Object Persistence

- Object IDs remain consistent across frames
- If an object disappears, it may still be tracked when it reappears (depending on occlusion duration)
- Bounding boxes are automatically updated based on the segmentation masks

### Performance

- Tracking is much faster than re-segmenting each frame
- SAM2 uses cached features from previous frames
- Typical performance: 20-30 FPS on GPU for tracking

## Mobile Client Integration

### Swift (iOS) Example

```swift
// 1. Initialize (first frame only)
let initRequest: [String: Any] = [
    "session_id": "my_session_\(UUID().uuidString)",
    "image": base64Image,
    "bounding_boxes": [[100, 100, 200, 200]]  // Only in first request!
]

// Send to /initialize
// Store session_id and object_ids from response

// 2. Track (subsequent frames)
for frame in videoStream {
    let trackRequest: [String: Any] = [
        "session_id": sessionId,  // From initialization
        "image": encodeFrame(frame)  // No bounding boxes!
    ]
    
    // Send to /track
    // Get updated bounding_boxes and object_ids
}
```

### Key Points for Mobile Clients

1. **Send bounding boxes only once** (in `/initialize`)
2. **Store the session_id** from the initialization response
3. **Send frames continuously** to `/track` without bounding boxes
4. **Use the returned bounding boxes** to display tracking results
5. **Object IDs remain consistent** - use them to maintain UI state

## Troubleshooting

### Objects Not Tracking

- Make sure you called `/initialize` first with bounding boxes
- Check that the bounding boxes are in the correct format: `[x0, y0, x1, y1]`
- Verify the session_id matches between initialize and track calls

### Tracking Drifts

- SAM2 is robust, but very fast motion or severe occlusions can cause drift
- Consider re-initializing if tracking quality degrades
- Use appropriate frame rate (2-5 FPS is usually sufficient)

### Multiple Objects

- You can track multiple objects by providing multiple bounding boxes in `/initialize`
- Each object gets a unique ID
- All objects are tracked simultaneously in each `/track` call

