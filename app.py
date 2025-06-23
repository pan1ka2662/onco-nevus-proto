import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


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

# --- DB & Login setup ---
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    uploads = db.relationship('Upload', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128), nullable=False)
    image_path = db.Column(db.String(256), nullable=False)
    diagnosis = db.Column(db.String(64), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    outputs = db.Column(db.PickleType)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- App factory ---
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key_here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
    db.init_app(app)
    login_manager.init_app(app)
    return app

app = create_app()

with app.app_context():
    db.create_all()

@app.route('/')
@login_required
def index():
    user_uploads = Upload.query.filter_by(user_id=current_user.id).order_by(Upload.date.desc()).limit(10).all()
    return render_template('index.html', history=user_uploads, current_user=current_user)

@app.route('/upload', methods=['POST'])
@login_required
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
            upload = Upload(
                filename=filename,
                image_path=f'/static/uploads/{filename}',
                diagnosis=diagnosis,
                date=datetime.now(),
                outputs=[output1, output2, output3],
                user_id=current_user.id
            )
            db.session.add(upload)
            db.session.commit()
            return jsonify({
                'diagnosis': diagnosis,
                'outputs': [output1, output2, output3],
                'image_path': upload.image_path
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Пользователь уже существует')
            return redirect(url_for('register'))
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Регистрация успешна! Войдите в систему.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Неверное имя пользователя или пароль')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)