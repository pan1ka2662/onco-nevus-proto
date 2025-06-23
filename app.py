import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json


app = Flask(__name__)

HISTORY_FILE = 'upload_history.json'

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        upload_history = json.load(f)
else:
    upload_history = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, checkpoint_path):
    if model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.6), 
            nn.Linear(1792, 1)
        )
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(num_features, 1)
        )
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_features, 1),
            nn.Flatten()
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

model1 = load_model('efficientnet_b4', 'tested_models/checkpoints_ENetb4/ENet.pth')
model2 = load_model('resnet50', 'tested_models/checkpoints_ResNet50/ResNet.pth')
model3 = load_model('convnext_small', 'tested_models/checkpoints_ConvNeXt_small/ConvNeXt.pth')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html', history=upload_history[-10:])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            upload_dir = 'static/uploads'
            os.makedirs(upload_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(upload_dir, filename)
            image = Image.open(file.stream).convert('RGB')
            image.save(filepath)
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output1 = model1(image_tensor).item()
                output2 = model2(image_tensor).item()
                output3 = model3(image_tensor).item()
            
            malignant = output1 > 0 or output2 > 0 or output3 > 0
            diagnosis = "злокачественная родинка" if malignant else "доброкачественная родинка"
            
            history_entry = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': filename,
                'diagnosis': diagnosis,
                'outputs': [output1, output2, output3],
                'image_path': f'/static/uploads/{filename}'
            }

            upload_history.append(history_entry)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(upload_history, f, indent=2)

            return jsonify({
                'diagnosis': diagnosis,
                'outputs': [output1, output2, output3],
                'image_path': history_entry['image_path']
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)