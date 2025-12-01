# SAM2 Mobile Server - Swift Integration Guide

This guide explains how to integrate the SAM2 Mobile Server with your Swift iOS/macOS application.

## Table of Contents
1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Swift Implementation](#swift-implementation)
4. [Complete Example](#complete-example)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

## Overview

The SAM2 Mobile Server provides REST API endpoints for:
- **Initialization**: Send the first frame with bounding boxes to start tracking
- **Tracking**: Send subsequent frames to track objects across frames
- **Health Check**: Verify server connectivity

### Server Requirements
- Server URL: `http://<SERVER_IP>:8080` (see [Network Setup Guide](NETWORK_SETUP_GUIDE.md) for details)
- Content-Type: `application/json`
- Image Format: Base64-encoded JPEG/PNG

### Important: Network Configuration

**If your server runs in WSL2**, you need to set up port forwarding to access it from other devices. See [NETWORK_SETUP_GUIDE.md](NETWORK_SETUP_GUIDE.md) for detailed instructions.

**Quick Summary:**
- WSL IP (e.g., `172.21.129.92`) is only accessible from the Windows host
- Use Windows network IP (e.g., `192.168.1.100`) from other devices
- Set up port forwarding: Windows IP:8080 → WSL IP:8080
- Configure Windows Firewall to allow port 8080

## API Endpoints

### 1. Health Check
**GET** `/health`

**Response:**
```json
{
  "status": "ok",
  "predictor_loaded": true
}
```

### 2. Initialize Session
**POST** `/initialize`

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "image": "base64_encoded_image",
  "bounding_boxes": [[x0, y0, x1, y1], ...],
  "object_ids": [1, 2, ...],  // Optional
  "include_masks": true       // Optional: request binary masks
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
  "masks": ["base64_mask_1", "base64_mask_2", ...]  // Present only if include_masks=true
}
```

**Mask Format:** Masks are base64-encoded binary arrays (0 = background, 1 = object) laid out in row-major order. See [Mask Handling](#mask-handling) for decoding instructions.

### 3. Track Frame
**POST** `/track`

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "image": "base64_encoded_image",
  "include_masks": true  // Optional: request binary masks for tracked objects
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], null, ...],  // null for lost objects
  "frame_index": 1,
  "frame_shape": [height, width],
  "masks": ["base64_mask_1", null, ...]  // Same ordering as object_ids (null for lost objects)
}
```

**Note:** Object IDs are always returned in the same order as initialized, maintaining consistency even when objects are temporarily lost. If an object is lost, its bounding box will be `null`, but the object ID is preserved. This ensures that:
- Object ID 1 always refers to the same object throughout the session
- Object ID 2 always refers to the same object throughout the session
- Lost objects return `null` for their bounding box but keep their ID

## Swift Implementation

### 1. Data Models

```swift
import Foundation
import UIKit

// MARK: - Request Models
struct InitializeRequest: Codable {
    let sessionId: String
    let image: String  // Base64 encoded
    let boundingBoxes: [[Double]]
    let objectIds: [Int]?
    let includeMasks: Bool?
    
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case image
        case boundingBoxes = "bounding_boxes"
        case objectIds = "object_ids"
        case includeMasks = "include_masks"
    }
}

struct TrackRequest: Codable {
    let sessionId: String
    let image: String  // Base64 encoded
    let includeMasks: Bool?
    
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case image
        case includeMasks = "include_masks"
    }
}

// MARK: - Response Models
struct HealthResponse: Codable {
    let status: String
    let predictorLoaded: Bool
    
    enum CodingKeys: String, CodingKey {
        case status
        case predictorLoaded = "predictor_loaded"
    }
}

struct InitializeResponse: Codable {
    let success: Bool
    let sessionId: String
    let objectIds: [Int]
    let boundingBoxes: [[Double]]
    let frameShape: [Int]
    let masks: [String?]?
    
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
    let boundingBoxes: [BoundingBox?]  // Optional - null for lost objects
    let frameIndex: Int
    let frameShape: [Int]
    let masks: [String?]?
    
    enum CodingKeys: String, CodingKey {
        case success
        case sessionId = "session_id"
        case objectIds = "object_ids"
        case boundingBoxes = "bounding_boxes"
        case frameIndex = "frame_index"
        case frameShape = "frame_shape"
        case masks
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        success = try container.decode(Bool.self, forKey: .success)
        sessionId = try container.decode(String.self, forKey: .sessionId)
        objectIds = try container.decode([Int].self, forKey: .objectIds)
        frameIndex = try container.decode(Int.self, forKey: .frameIndex)
        frameShape = try container.decode([Int].self, forKey: .frameShape)
        masks = try container.decodeIfPresent([String?].self, forKey: .masks)
        
        // Handle optional bounding boxes (null values)
        if let bboxArray = try? container.decode([OptionalBoundingBox].self, forKey: .boundingBoxes) {
            boundingBoxes = bboxArray.map { $0.bbox }
        } else {
            boundingBoxes = []
        }
    }
}

// Helper to decode optional bounding boxes
private struct OptionalBoundingBox: Codable {
    let bbox: BoundingBox?
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            bbox = nil
        } else {
            let array = try container.decode([Double].self)
            bbox = BoundingBox(from: array)
        }
    }
}

// MARK: - Bounding Box Model
struct BoundingBox {
    let x0: Double
    let y0: Double
    let x1: Double
    let y1: Double
    
    var asArray: [Double] {
        return [x0, y0, x1, y1]
    }
    
    init(x0: Double, y0: Double, x1: Double, y1: Double) {
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }
    
    init(from array: [Double]) {
        guard array.count == 4 else {
            fatalError("Bounding box array must have 4 elements")
        }
        self.x0 = array[0]
        self.y0 = array[1]
        self.x1 = array[2]
        self.y1 = array[3]
    }
}
```

### Mask Handling Utilities

Masks (if requested) are base64-encoded binary arrays that match `frame_shape` and align with `objectIds`. Use the helpers below to decode masks and convert them into overlays for annotation.

```swift
import Foundation
import UIKit

typealias BinaryMask = [[UInt8]]

/// Decode a base64 mask string into a 2D UInt8 array using frame dimensions
func decodeMask(_ encodedMask: String?, frameShape: [Int]) -> BinaryMask? {
    guard let encodedMask = encodedMask,
          frameShape.count == 2,
          let data = Data(base64Encoded: encodedMask) else {
        return nil
    }
    
    let height = frameShape[0]
    let width = frameShape[1]
    let bytes = [UInt8](data)
    guard bytes.count == height * width else {
        return nil
    }
    
    var mask: BinaryMask = []
    mask.reserveCapacity(height)
    for row in 0..<height {
        let start = row * width
        let end = start + width
        mask.append(Array(bytes[start..<end]))
    }
    return mask
}

/// Create a semi-transparent overlay from a binary mask
func createMaskOverlay(mask: BinaryMask, color: UIColor, alpha: CGFloat = 0.4) -> UIImage? {
    guard let firstRow = mask.first else { return nil }
    let height = mask.count
    let width = firstRow.count
    
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    var pixelBytes = [UInt8](repeating: 0, count: height * bytesPerRow)
    
    var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
    color.getRed(&r, green: &g, blue: &b, alpha: &a)
    
    for row in 0..<height {
        for col in 0..<width {
            guard mask[row][col] == 1 else { continue }
            let offset = row * bytesPerRow + col * bytesPerPixel
            pixelBytes[offset] = UInt8(r * 255)
            pixelBytes[offset + 1] = UInt8(g * 255)
            pixelBytes[offset + 2] = UInt8(b * 255)
            pixelBytes[offset + 3] = UInt8(alpha * 255)
        }
    }
    
    guard let provider = CGDataProvider(data: Data(pixelBytes) as CFData),
          let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: bytesPerPixel * 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
          ) else {
        return nil
    }
    
    return UIImage(cgImage: cgImage)
}
```

### 2. Image Encoding Utility

```swift
import UIKit

extension UIImage {
    /// Convert UIImage to base64 encoded JPEG string
    func toBase64(quality: CGFloat = 0.85) -> String? {
        guard let imageData = self.jpegData(compressionQuality: quality) else {
            return nil
        }
        return imageData.base64EncodedString()
    }
    
    /// Convert UIImage to base64 encoded PNG string
    func toBase64PNG() -> String? {
        guard let imageData = self.pngData() else {
            return nil
        }
        return imageData.base64EncodedString()
    }
}
```

### 3. SAM2 Client Service

```swift
import Foundation

enum SAM2Error: Error {
    case invalidURL
    case invalidImage
    case networkError(Error)
    case serverError(Int, String)
    case decodingError(Error)
    case invalidResponse
}

class SAM2Client {
    private let baseURL: String
    private let session: URLSession
    
    init(baseURL: String) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30.0
        config.timeoutIntervalForResource = 60.0
        self.session = URLSession(configuration: config)
    }
    
    // MARK: - Health Check
    func checkHealth() async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/health") else {
            throw SAM2Error.invalidURL
        }
        
        let (data, response) = try await session.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw SAM2Error.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw SAM2Error.serverError(httpResponse.statusCode, errorMessage)
        }
        
        do {
            let healthResponse = try JSONDecoder().decode(HealthResponse.self, from: data)
            return healthResponse
        } catch {
            throw SAM2Error.decodingError(error)
        }
    }
    
    // MARK: - Initialize Session
    func initializeSession(
        sessionId: String,
        image: UIImage,
        boundingBoxes: [BoundingBox],
        objectIds: [Int]? = nil,
        requestMasks: Bool = false
    ) async throws -> InitializeResponse {
        guard let imageBase64 = image.toBase64() else {
            throw SAM2Error.invalidImage
        }
        
        // Generate object IDs if not provided
        let finalObjectIds = objectIds ?? Array(1...boundingBoxes.count)
        
        let request = InitializeRequest(
            sessionId: sessionId,
            image: imageBase64,
            boundingBoxes: boundingBoxes.map { $0.asArray },
            objectIds: finalObjectIds,
            includeMasks: requestMasks ? true : nil
        )
        
        guard let url = URL(string: "\(baseURL)/initialize") else {
            throw SAM2Error.invalidURL
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            throw SAM2Error.decodingError(error)
        }
        
        let (data, response) = try await session.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw SAM2Error.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw SAM2Error.serverError(httpResponse.statusCode, errorMessage)
        }
        
        do {
            let initResponse = try JSONDecoder().decode(InitializeResponse.self, from: data)
            return initResponse
        } catch {
            throw SAM2Error.decodingError(error)
        }
    }
    
    // MARK: - Track Frame
    func trackFrame(sessionId: String, image: UIImage, requestMasks: Bool = false) async throws -> TrackResponse {
        guard let imageBase64 = image.toBase64() else {
            throw SAM2Error.invalidImage
        }
        
        let request = TrackRequest(sessionId: sessionId, image: imageBase64, includeMasks: requestMasks ? true : nil)
        
        guard let url = URL(string: "\(baseURL)/track") else {
            throw SAM2Error.invalidURL
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            throw SAM2Error.decodingError(error)
        }
        
        let (data, response) = try await session.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw SAM2Error.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw SAM2Error.serverError(httpResponse.statusCode, errorMessage)
        }
        
        do {
            let trackResponse = try JSONDecoder().decode(TrackResponse.self, from: data)
            return trackResponse
        } catch {
            throw SAM2Error.decodingError(error)
        }
    }
}
```

## Complete Example

### Usage in ViewController

```swift
import UIKit

class TrackingViewController: UIViewController {
    private let client: SAM2Client
    private var sessionId: String
    private var isInitialized = false
    
    init(serverURL: String) {
        self.client = SAM2Client(baseURL: serverURL)
        self.sessionId = UUID().uuidString
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        checkServerHealth()
    }
    
    // MARK: - Server Health Check
    private func checkServerHealth() {
        Task {
            do {
                let health = try await client.checkHealth()
                print("Server is healthy: \(health.predictorLoaded)")
            } catch {
                print("Health check failed: \(error)")
                // Show error to user
                DispatchQueue.main.async {
                    self.showError("Cannot connect to server: \(error.localizedDescription)")
                }
            }
        }
    }
    
    // MARK: - Initialize Tracking
    func initializeTracking(image: UIImage, boundingBoxes: [BoundingBox], requestMasks: Bool = false) {
        Task {
            do {
                let response = try await client.initializeSession(
                    sessionId: sessionId,
                    image: image,
                    boundingBoxes: boundingBoxes,
                    requestMasks: requestMasks
                )
                
                DispatchQueue.main.async {
                    self.isInitialized = true
                    print("Initialized with \(response.objectIds.count) objects")
                    print("Bounding boxes: \(response.boundingBoxes)")
                    
                    if requestMasks {
                        self.logMasks(response.masks, frameShape: response.frameShape)
                    }
                }
            } catch {
                print("Initialization failed: \(error)")
                DispatchQueue.main.async {
                    self.showError("Failed to initialize: \(error.localizedDescription)")
                }
            }
        }
    }
    
    // MARK: - Track Frame
    func trackFrame(image: UIImage, requestMasks: Bool = false) {
        guard isInitialized else {
            print("Session not initialized. Call initializeTracking first.")
            return
        }
        
        Task {
            do {
                let response = try await client.trackFrame(
                    sessionId: sessionId,
                    image: image,
                    requestMasks: requestMasks
                )
                
                DispatchQueue.main.async {
                    print("Frame \(response.frameIndex) tracked")
                    print("Object IDs: \(response.objectIds)")
                    print("Bounding boxes: \(response.boundingBoxes)")
                    
                    // Update UI with tracking results
                    self.updateTrackingResults(response)
                }
            } catch {
                print("Tracking failed: \(error)")
                DispatchQueue.main.async {
                    self.showError("Tracking failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    private func updateTrackingResults(_ response: TrackResponse) {
        for (index, objectId) in response.objectIds.enumerated() {
            let bbox = index < response.boundingBoxes.count ? response.boundingBoxes[index] : nil
            if let bbox = bbox {
                // Object is tracked - update UI with bounding box
                print("Object \(objectId) is at: \(bbox.asArray)")
                // Draw bounding box on UI
            } else {
                // Object is lost - hide or mark as lost
                print("Object \(objectId) is lost (not tracked in this frame)")
                // Hide bounding box or show "lost" indicator
            }
            
            let maskString: String?
            if let masks = response.masks, index < masks.count {
                maskString = masks[index]
            } else {
                maskString = nil
            }
            
            if let mask = decodeMask(maskString, frameShape: response.frameShape) {
                print("Mask for object \(objectId) decoded (\(mask.count) x \(mask.first?.count ?? 0))")
                // Convert to overlay: let overlay = createMaskOverlay(mask: mask, color: .systemBlue)
                // Display overlay on top of your frame/image view
            }
        }
    }
    
    private func logMasks(_ masks: [String?]?, frameShape: [Int]) {
        guard let masks = masks else { return }
        for (index, maskString) in masks.enumerated() {
            if let mask = decodeMask(maskString, frameShape: frameShape) {
                print("Decoded mask for init object \(index + 1) (\(mask.count) x \(mask.first?.count ?? 0))")
            }
        }
    }
    
    private func showError(_ message: String) {
        let alert = UIAlertController(
            title: "Error",
            message: message,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}
```

### Camera Integration Example

```swift
import AVFoundation

class CameraTrackingViewController: UIViewController {
    private let client: SAM2Client
    private var sessionId: String
    private var isInitialized = false
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    
    init(serverURL: String) {
        self.client = SAM2Client(baseURL: serverURL)
        self.sessionId = UUID().uuidString
        super.init(nibName: nil, bundle: nil)
    }
    
    // MARK: - Camera Setup
    func setupCamera() {
        let session = AVCaptureSession()
        session.sessionPreset = .medium
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            
            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "cameraQueue"))
            
            if session.canAddOutput(output) {
                session.addOutput(output)
                self.videoOutput = output
            }
            
            self.captureSession = session
        } catch {
            print("Camera setup error: \(error)")
        }
    }
    
    // MARK: - Initialize with First Frame
    func initializeWithFirstFrame(_ image: UIImage, boundingBoxes: [BoundingBox], requestMasks: Bool = true) {
        Task {
            do {
                let response = try await client.initializeSession(
                    sessionId: sessionId,
                    image: image,
                    boundingBoxes: boundingBoxes,
                    requestMasks: requestMasks
                )
                
                DispatchQueue.main.async {
                    self.isInitialized = true
                    print("Initialized successfully")
                    if requestMasks {
                        self.logMasks(response.masks, frameShape: response.frameShape)
                    }
                }
            } catch {
                print("Initialization error: \(error)")
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraTrackingViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isInitialized else { return }
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        // Track frame (throttle to 2 FPS)
        trackFrame(image: uiImage, requestMasks: true)
    }
    
    private var lastTrackTime: Date = Date()
    private let trackInterval: TimeInterval = 0.5 // 2 FPS
    
    private func trackFrame(image: UIImage, requestMasks: Bool) {
        let now = Date()
        guard now.timeIntervalSince(lastTrackTime) >= trackInterval else { return }
        lastTrackTime = now
        
        Task {
            do {
                let response = try await client.trackFrame(sessionId: sessionId, image: image, requestMasks: requestMasks)
                DispatchQueue.main.async {
                    // Update UI with tracking results
                    self.updateTrackingResults(response)
                }
            } catch {
                print("Tracking error: \(error)")
            }
        }
    }
    
    private func updateTrackingResults(_ response: TrackResponse) {
        for (index, objectId) in response.objectIds.enumerated() {
            let bbox = index < response.boundingBoxes.count ? response.boundingBoxes[index] : nil
            if let bbox = bbox {
                print("Camera object \(objectId) bbox: \(bbox.asArray)")
            } else {
                print("Camera object \(objectId) lost")
            }
            
            let maskString: String?
            if let masks = response.masks, index < masks.count {
                maskString = masks[index]
            } else {
                maskString = nil
            }
            
            if let mask = decodeMask(maskString, frameShape: response.frameShape),
               let overlay = createMaskOverlay(mask: mask, color: .systemGreen, alpha: 0.3) {
                // Display overlay on your preview layer / UIImageView
                print("Overlay ready for object \(objectId) (\(overlay.size))")
            }
        }
    }
    
    private func logMasks(_ masks: [String?]?, frameShape: [Int]) {
        guard let masks = masks else { return }
        for (index, maskString) in masks.enumerated() {
            if let mask = decodeMask(maskString, frameShape: frameShape) {
                print("Camera init mask \(index + 1) decoded (\(mask.count)x\(mask.first?.count ?? 0))")
            }
        }
    }
}
```

## Error Handling

```swift
func handleError(_ error: SAM2Error) {
    switch error {
    case .invalidURL:
        print("Invalid server URL")
    case .invalidImage:
        print("Failed to encode image")
    case .networkError(let underlyingError):
        print("Network error: \(underlyingError.localizedDescription)")
    case .serverError(let statusCode, let message):
        print("Server error \(statusCode): \(message)")
    case .decodingError(let underlyingError):
        print("Decoding error: \(underlyingError.localizedDescription)")
    case .invalidResponse:
        print("Invalid response from server")
    }
}
```

## Best Practices

### 1. Image Compression
```swift
// Compress images before sending to reduce payload size
func compressImage(_ image: UIImage, maxSize: Int = 1024) -> UIImage? {
    var compression: CGFloat = 0.9
    var imageData = image.jpegData(compressionQuality: compression)
    
    while let data = imageData, data.count > maxSize * 1024 && compression > 0.1 {
        compression -= 0.1
        imageData = image.jpegData(compressionQuality: compression)
    }
    
    return imageData.flatMap { UIImage(data: $0) }
}
```

### 2. Frame Rate Throttling
```swift
// Throttle tracking to 2 FPS
private var lastTrackTime: Date = Date()
private let trackInterval: TimeInterval = 0.5

func shouldTrackFrame() -> Bool {
    let now = Date()
    guard now.timeIntervalSince(lastTrackTime) >= trackInterval else {
        return false
    }
    lastTrackTime = now
    return true
}
```

### 3. Session Management
```swift
// Reset session when needed
func resetSession() {
    sessionId = UUID().uuidString
    isInitialized = false
}
```

### 4. Network Configuration
```swift
// For local network access (WSL/development)
// Add to Info.plist:
// <key>NSAppTransportSecurity</key>
// <dict>
//     <key>NSAllowsLocalNetworking</key>
//     <true/>
// </dict>
```

## Testing

### Test Server Connection
```swift
let client = SAM2Client(baseURL: "http://172.21.129.92:8080")

Task {
    do {
        let health = try await client.checkHealth()
        print("✓ Server is healthy")
    } catch {
        print("✗ Connection failed: \(error)")
    }
}
```

## Troubleshooting

1. **Connection Refused**: Check server IP and port, ensure server is running
2. **Timeout**: Increase timeout intervals in URLSessionConfiguration
3. **Image Encoding Fails**: Ensure UIImage is valid and not nil
4. **CORS Issues**: Server should have CORS enabled (already configured)
5. **Large Payload**: Compress images before encoding to base64

## Additional Resources

- Server API Documentation: See `SAM_MOBILE_SERVER_README.md`
- Python Test Client: See `test_mobile_server.py` for reference implementation

