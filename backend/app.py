import random
import os
import glob
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import shutil
from torchvision import transforms
from model import Model
from PredictingCode import predict, ValidationDataset

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Directory to store uploaded files
UPLOAD_DIRECTORY = "../uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Model and transformation parameters
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the model
model = pickle.load(open(os.path.join(os.getcwd(), 'saved_models', './OG_Pretrained_Test_model_epoch_20+10.sav'), 'rb'))
model.eval()

def clear_upload_directory():
    """Remove all files in the upload directory."""
    for file in glob.glob(os.path.join(UPLOAD_DIRECTORY, '*')):
        os.remove(file)
def clear_save_directory(save_directory):
    """Remove all files in the extracted directory."""
    for file in glob.glob(os.path.join(save_directory, '*')):
        os.remove(file)


def process_video(file_path):
    """Process the video file and return predictions with confidence."""
    try:
        path_to_videos = [file_path]
        save_directory = "extracted_frames"
        clear_save_directory(save_directory)
        # Assuming ValidationDataset is defined elsewhere in your code
        video_dataset = ValidationDataset(path_to_videos, save_directory, sequence_length=15, transform=train_transforms)
        
        results = []

        # Process frames and make predictions
        for i in range(len(video_dataset)):
            # Assuming predict is defined elsewhere in your code
            prediction_result = predict(model, video_dataset[i], './')
            prediction, confidence = prediction_result
            label = "REAL" if prediction == 1 else "FAKE"
            results.append({"label": label, "confidence": confidence})

        # Get all frame paths from extracted_frames directory
        if os.path.exists(save_directory):
            frame_paths = [os.path.join(save_directory, f) for f in os.listdir(save_directory) if os.path.isfile(os.path.join(save_directory, f))]
        else:
            frame_paths = []

        # Select random 5 frame paths if available
        random_frames = random.sample(frame_paths, min(5, len(frame_paths)))

        return results, random_frames
    except Exception as e:
        raise RuntimeError(f"Error during video processing: {str(e)}")


@app.route('/extracted_frames/<filename>')
def serve_frame(filename):
    """Serve the extracted frame image."""
    extracted_frames_path = os.path.join(os.getcwd(), 'extracted_frames', filename)
    if os.path.exists(extracted_frames_path):
        return send_from_directory(os.path.join(os.getcwd(), 'extracted_frames'), filename)
    else:
        return jsonify({"error": "File not found"}), 404


@app.route('/process', methods=['POST'])
def process_data():
    """Process JSON data."""
    data = request.json
    if 'name' not in data:
        return jsonify({"error": "Missing 'name' in JSON data"}), 400
    result = {"message": f"Hello {data['name']}"}
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, process the video, and return predictions."""
    # Clear the upload directory before saving the new file
    clear_upload_directory()

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if the file is a video
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({"error": "File type not allowed. Please upload a video file."}), 400
    
    # Save the file to the specified directory
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    
    # Process the video file and get predictions and random frames
    predictions, random_frames = process_video(file_path)

    # Cleanup extracted frames directory after processing
    
    return jsonify({
        "message": "Video processed successfully",
        "predictions": predictions,
        "file_path": file_path,
        "random_frames": random_frames  # Return paths of random extracted frames
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
