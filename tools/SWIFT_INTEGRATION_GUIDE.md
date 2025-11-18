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
  "object_ids": [1, 2, ...]  // Optional
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

### 3. Track Frame
**POST** `/track`

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "unique_session_id",
  "object_ids": [1, 2, ...],
  "bounding_boxes": [[x0, y0, x1, y1], null, ...],
  "frame_index": 1
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
    
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case image
        case boundingBoxes = "bounding_boxes"
        case objectIds = "object_ids"
    }
}

struct TrackRequest: Codable {
    let sessionId: String
    let image: String  // Base64 encoded
    
    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case image
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
    
    enum CodingKeys: String, CodingKey {
        case success
        case sessionId = "session_id"
        case objectIds = "object_ids"
        case boundingBoxes = "bounding_boxes"
        case frameShape = "frame_shape"
    }
}

struct TrackResponse: Codable {
    let success: Bool
    let sessionId: String
    let objectIds: [Int]
    let boundingBoxes: [BoundingBox?]  // Optional - null for lost objects
    let frameIndex: Int
    
    enum CodingKeys: String, CodingKey {
        case success
        case sessionId = "session_id"
        case objectIds = "object_ids"
        case boundingBoxes = "bounding_boxes"
        case frameIndex = "frame_index"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        success = try container.decode(Bool.self, forKey: .success)
        sessionId = try container.decode(String.self, forKey: .sessionId)
        objectIds = try container.decode([Int].self, forKey: .objectIds)
        frameIndex = try container.decode(Int.self, forKey: .frameIndex)
        
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
        objectIds: [Int]? = nil
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
            objectIds: finalObjectIds
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
    func trackFrame(sessionId: String, image: UIImage) async throws -> TrackResponse {
        guard let imageBase64 = image.toBase64() else {
            throw SAM2Error.invalidImage
        }
        
        let request = TrackRequest(sessionId: sessionId, image: imageBase64)
        
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
    func initializeTracking(image: UIImage, boundingBoxes: [BoundingBox]) {
        Task {
            do {
                let response = try await client.initializeSession(
                    sessionId: sessionId,
                    image: image,
                    boundingBoxes: boundingBoxes
                )
                
                DispatchQueue.main.async {
                    self.isInitialized = true
                    print("Initialized with \(response.objectIds.count) objects")
                    print("Bounding boxes: \(response.boundingBoxes)")
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
    func trackFrame(image: UIImage) {
        guard isInitialized else {
            print("Session not initialized. Call initializeTracking first.")
            return
        }
        
        Task {
            do {
                let response = try await client.trackFrame(
                    sessionId: sessionId,
                    image: image
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
        // Handle tracking results with consistent object IDs
        // Note: boundingBoxes array may contain nil for lost objects
        
        for (index, objectId) in response.objectIds.enumerated() {
            if let bbox = response.boundingBoxes[index] {
                // Object is tracked - update UI with bounding box
                print("Object \(objectId) is at: \(bbox.asArray)")
                // Draw bounding box on UI
            } else {
                // Object is lost - hide or mark as lost
                print("Object \(objectId) is lost (not tracked in this frame)")
                // Hide bounding box or show "lost" indicator
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
    func initializeWithFirstFrame(_ image: UIImage, boundingBoxes: [BoundingBox]) {
        Task {
            do {
                let response = try await client.initializeSession(
                    sessionId: sessionId,
                    image: image,
                    boundingBoxes: boundingBoxes
                )
                
                DispatchQueue.main.async {
                    self.isInitialized = true
                    print("Initialized successfully")
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
        trackFrame(image: uiImage)
    }
    
    private var lastTrackTime: Date = Date()
    private let trackInterval: TimeInterval = 0.5 // 2 FPS
    
    private func trackFrame(image: UIImage) {
        let now = Date()
        guard now.timeIntervalSince(lastTrackTime) >= trackInterval else { return }
        lastTrackTime = now
        
        Task {
            do {
                let response = try await client.trackFrame(sessionId: sessionId, image: image)
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
        // Update your UI here
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

