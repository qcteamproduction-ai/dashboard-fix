from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class YOLODetectionSystem:
    def __init__(self, model_path="testing.pt"):
        self.model = YOLO(model_path)
        self.camera = None
        self.is_running = False
        self.current_frame = None
        
        # Print available class names for debugging
        print("Available model classes:")
        for i, class_name in enumerate(self.model.names.values()):
            print(f"  {i}: {class_name}")
        
        self.detection_results = {
            'has_defects': False,
            'detections': [],
            'fps': 0,
            'stats': {'total': 0, 'pass': 0, 'ng': 0},
            'status': 'PASS'  # PASS, NG
        }
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def process_detections(self, results, frame):
        """Process YOLO detection results"""
        defects_detected = False
        detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Debug: Print all detected classes
            print(f"Total detections: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"Class_{class_id}"
                
                # Debug: Print detection info
                print(f"Detection {i}: Class='{class_name}', Confidence={confidence:.2f}")
                
                detection_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox.tolist()
                }
                detections.append(detection_info)
                
                # Check for defects (SG-defect objects)
                defect_classes = ['sg-defect', 'sg_defect', 'sgdefect', 'sg defect', 'defect']
                is_defect = (class_name.lower() in defect_classes or 
                            'defect' in class_name.lower() or
                            'sg' in class_name.lower())
                
                if is_defect:
                    defects_detected = True
                    print(f"SG-DEFECT DETECTED: {class_name}")
                
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Color coding based on object type
                if is_defect:
                    color = (0, 0, 255)  # Red for SG-defect
                    status = "SG-DEFECT"
                else:
                    color = (0, 255, 0)  # Green for other objects
                    status = class_name.upper()
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{status} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return defects_detected, detections, frame
    
    def determine_status(self, defects_detected):
        """Determine system status based on detection logic"""
        if defects_detected:
            return "NG"
        else:
            return "PASS"
    
    def update_statistics(self, status):
        """Update production statistics"""
        current_time = time.time()
        if not hasattr(self, 'last_detection_time'):
            self.last_detection_time = 0
        
        # Count every 3 seconds to avoid rapid counting
        if current_time - self.last_detection_time > 3.0:
            self.detection_results['stats']['total'] += 1
            if status == 'NG':
                self.detection_results['stats']['ng'] += 1
            elif status == 'PASS':
                self.detection_results['stats']['pass'] += 1
            self.last_detection_time = current_time
    
    def calculate_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.detection_results['fps'] = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def generate_frames(self):
        """Generate video frames with detections"""
        while self.is_running and self.camera is not None:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            try:
                results = self.model(frame, conf=0.5)
                defects_detected, detections, processed_frame = self.process_detections(results, frame)
                
                # Determine status based on detection logic
                current_status = self.determine_status(defects_detected)
                
                self.detection_results['has_defects'] = defects_detected
                self.detection_results['detections'] = detections
                self.detection_results['status'] = current_status
                
                self.update_statistics(current_status)
                self.calculate_fps()
                
                # Status display logic
                if current_status == "PASS":
                    status_color = (0, 255, 0)  # Green
                    status_text_main = "PASS"
                    status_text_detail = "No SG-defects detected"
                elif current_status == "NG":
                    status_color = (0, 0, 255)  # Red
                    status_text_main = "NG"
                    status_text_detail = "SG-defect detected!"
                
                # Display status on frame
                fps_text = f"FPS: {self.detection_results['fps']}"
                
                # Create overlay for status display
                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (5, 5), (650, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
                
                # Display FPS
                cv2.putText(processed_frame, fps_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display main status
                cv2.putText(processed_frame, f"Status: {status_text_main}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Display detail
                cv2.putText(processed_frame, status_text_detail, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                self.current_frame = processed_frame.copy()
                
                # Encode frame to base64 for Socket.IO
                ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit detection results and frame
                    socketio.emit('detection_result', {
                        'status': current_status,
                        'defects_detected': defects_detected,
                        'pass_count': self.detection_results['stats']['pass'],
                        'ng_count': self.detection_results['stats']['ng'],
                        'total_count': self.detection_results['stats']['total'],
                        'ng_rate': round((self.detection_results['stats']['ng'] / 
                                       max(1, self.detection_results['stats']['total']) * 100), 1)
                    })
                    socketio.emit('video_frame', {'frame': frame_bytes})
            
            except Exception as e:
                print(f"Detection error: {e}")
                self.current_frame = frame.copy()
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_detection(self):
        """Start detection system"""
        if self.initialize_camera():
            self.is_running = True
            threading.Thread(target=self.generate_frames, daemon=True).start()
            return True
        return False
    
    def stop_detection(self):
        """Stop detection system"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def restart_detection(self):
        """Restart detection system"""
        self.stop_detection()
        time.sleep(0.5)
        return self.start_detection()

# Initialize detection system
detector = YOLODetectionSystem()

@app.route('/')
def index():
    """Main page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oppo Phone Detection Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        .video-container {
            position: relative;
            width: 100%;
            height: 0;
            padding-bottom: 75%;
        }
        .video-feed {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .detection-banner {
            transition: background-color 0.3s ease;
        }
        .pass-bg {
            background: linear-gradient(135deg, #10B981, #059669);
            color: white;
        }
        .ng-bg {
            background: linear-gradient(135deg, #EF4444, #DC2626);
            color: white;
        }
        .status-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-pass { background-color: #10B981; }
        .status-ng { background-color: #EF4444; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-6">
        <div id="detectionBanner" class="detection-banner rounded-lg shadow-lg mb-6 py-8 text-center pass-bg">
            <div class="flex items-center justify-center mb-2">
                <div id="statusIndicator" class="status-indicator status-pass"></div>
                <h1 id="detectionStatus" class="text-5xl font-bold">PASS</h1>
            </div>
            <p id="detectionDetail" class="text-xl opacity-90">No SG-defects detected</p>
        </div>

        <div class="flex flex-col lg:flex-row gap-6">
            <div class="flex-1">
                <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                    <div class="bg-gray-50 px-4 py-2 border-b">
                        <h3 class="text-lg font-semibold text-gray-700">Live Camera Feed</h3>
                        <p class="text-sm text-gray-500">Real-time SG-defect detection</p>
                    </div>
                    <div class="video-container">
                        <img id="videoFeed" class="video-feed" src="" alt="Video Feed">
                    </div>
                </div>
            </div>

            <div class="w-full lg:w-80">
                <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-6">System Controls</h2>
                    
                    <div class="flex flex-col space-y-4">
                        <button id="startBtn" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition">
                            Start Detection
                        </button>
                        <button id="stopBtn" class="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition" disabled>
                            Stop Detection
                        </button>
                        <button id="restartBtn" class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition" disabled>
                            Restart System
                        </button>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Detection Status</h2>
                    <div class="space-y-3">
                        <div class="flex justify-between">
                            <span class="font-medium">SG-defect Found:</span>
                            <span id="defectStatus" class="font-bold text-gray-500">No</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">Current Status:</span>
                            <span id="currentStatus" class="font-bold text-green-600">PASS</span>
                        </div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Production Statistics</h2>
                    <div class="space-y-3">
                        <div class="flex justify-between">
                            <span class="font-medium">Pass Count:</span>
                            <span id="passCount" class="font-bold text-green-600">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">NG Count:</span>
                            <span id="ngCount" class="font-bold text-red-600">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">Total:</span>
                            <span id="totalCount" class="font-bold">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">NG Rate:</span>
                            <span id="ngRate" class="font-bold">0%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const videoFeed = document.getElementById('videoFeed');
        const detectionBanner = document.getElementById('detectionBanner');
        const detectionStatus = document.getElementById('detectionStatus');
        const detectionDetail = document.getElementById('detectionDetail');
        const statusIndicator = document.getElementById('statusIndicator');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const restartBtn = document.getElementById('restartBtn');
        const defectStatus = document.getElementById('defectStatus');
        const currentStatus = document.getElementById('currentStatus');
        const passCount = document.getElementById('passCount');
        const ngCount = document.getElementById('ngCount');
        const totalCount = document.getElementById('totalCount');
        const ngRate = document.getElementById('ngRate');

        function updateButtonStates(isRunning) {
            startBtn.disabled = isRunning;
            stopBtn.disabled = !isRunning;
            restartBtn.disabled = !isRunning;
        }

        function updateStatusDisplay(status, defectsDetected) {
            // Update status indicator and banner
            if (status === 'PASS') {
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center pass-bg';
                statusIndicator.className = 'status-indicator status-pass';
                detectionStatus.textContent = 'PASS';
                detectionDetail.textContent = 'No SG-defects detected';
            } else if (status === 'NG') {
                detectionBanner.className = 'detection-banner rounded-lg shadow-lg mb-6 py-8 text-center ng-bg';
                statusIndicator.className = 'status-indicator status-ng';
                detectionStatus.textContent = 'NG';
                detectionDetail.textContent = 'SG-defect detected!';
            }

            // Update status details
            defectStatus.textContent = defectsDetected ? 'Yes' : 'No';
            defectStatus.className = defectsDetected ? 'font-bold text-red-600' : 'font-bold text-gray-500';
            
            currentStatus.textContent = status;
            if (status === 'PASS') {
                currentStatus.className = 'font-bold text-green-600';
            } else if (status === 'NG') {
                currentStatus.className = 'font-bold text-red-600';
            }
        }

        startBtn.addEventListener('click', () => {
            socket.emit('start_detection');
        });

        stopBtn.addEventListener('click', () => {
            socket.emit('stop_detection');
        });

        restartBtn.addEventListener('click', () => {
            socket.emit('restart_system');
        });

        socket.on('connect', () => {
            console.log('Connected to server');
        });

        socket.on('system_status', (data) => {
            updateButtonStates(data.is_running);
            if (!data.is_running) {
                videoFeed.src = '';
                updateStatusDisplay('PASS', false);
            }
        });

        socket.on('video_frame', (data) => {
            videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
        });

        socket.on('detection_result', (data) => {
            updateStatusDisplay(data.status, data.defects_detected);
            
            passCount.textContent = data.pass_count;
            ngCount.textContent = data.ng_count;
            totalCount.textContent = data.total_count;
            ngRate.textContent = data.ng_rate + '%';
        });
    </script>
</body>
</html>
    '''

@socketio.on('start_detection')
def handle_start_detection():
    if detector.start_detection():
        emit('system_status', {'is_running': True})
    else:
        emit('system_status', {'is_running': False, 'message': 'Failed to initialize camera'})

@socketio.on('stop_detection')
def handle_stop_detection():
    detector.stop_detection()
    emit('system_status', {'is_running': False})

@socketio.on('restart_system')
def handle_restart_system():
    if detector.restart_detection():
        emit('system_status', {'is_running': True})
    else:
        emit('system_status', {'is_running': False, 'message': 'Failed to restart detection'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
