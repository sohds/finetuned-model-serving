import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN

# 잘린 이미지 로드 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 클래스 인덱스 생성
        for idx, class_name in enumerate(sorted(os.listdir(directory))):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(directory, class_name)
            
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class PneumoniaModelModified(nn.Module):
    def __init__(self, num_classes=4, input_features=1536):
        super(PneumoniaModelModified, self).__init__()
        
        # 입력으로 들어오는 데이터는 이미:
        # 1. EfficientNet으로 특징 추출 완료
        # 2. 평균 풀링 적용 완료
        # 3. 평탄화(flatten) 완료
        # 4. 크기: [batch_size, 1536]
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),  # 1536 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),            # 512 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)     # 256 -> num_classes
        )

    def forward(self, x):
        # x의 예상 shape: [batch_size, 1536]
        x = self.classifier(x)
        return x

def setup_directories(data_directory, writable_directory, train_directory, val_directory):
    """데이터 디렉토리 설정 및 초기화"""
    # 기존 디렉토리 삭제 후 재생성
    for dir_path in [writable_directory, train_directory, val_directory]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # 데이터 복사
    for class_name in os.listdir(data_directory):
        class_directory = os.path.join(data_directory, class_name)
        writable_class_directory = os.path.join(writable_directory, class_name)
        os.makedirs(writable_class_directory, exist_ok=True)
        
        for image in os.listdir(class_directory):
            src_path = os.path.join(class_directory, image)
            dest_path = os.path.join(writable_class_directory, image)
            shutil.copy(src_path, dest_path)

def prepare_data_splits(writable_directory, train_dir, val_dir, test_size=0.2):
    """데이터를 train과 validation으로 분할"""
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    for class_name in os.listdir(writable_directory):
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir)
        os.makedirs(val_class_dir)
        
        class_dir = os.path.join(writable_directory, class_name)
        images = os.listdir(class_dir)
        
        train_images, val_images = train_test_split(
            images, test_size=test_size, random_state=42
        )
        
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
        
        print(f"{class_name} - Train: {len(train_images)}, Val: {len(val_images)}")

def create_feature_extractor():
    """EfficientNetB3 특징 추출기 생성"""
    model = models.efficientnet_b3(pretrained=True)
    
    # 초기 레이어를 동결
    for param in model.parameters():
        param.requires_grad = False
    
    # 평균 풀링 레이어까지만 사용
    features_model = nn.Sequential(*list(model.children())[:-1])
    features_model.eval()
    return features_model

def extract_features(data_dir, feature_extractor, device="cuda"):
    """디렉토리에서 이미지 특징 추출"""
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    labels = []
    feature_extractor.to(device)
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_features = feature_extractor(images)
            batch_features = batch_features.view(batch_features.size(0), -1)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    X = np.concatenate(features)
    y = np.array(labels)
    
    return X, y

def apply_sampling_strategies(X, y, random_state=42):
    """ADASYN 샘플링 적용"""
    ada = ADASYN(random_state=random_state)
    X_resampled, y_resampled = ada.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    """모델 학습"""
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.3f}, Train Acc: {train_acc:.3f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f}, Val Acc: {val_acc:.3f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, test_loader, device="cuda"):
    """모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # 클래스 이름 매핑
    class_names = ['COVID-19', 'Bacterial Pneumonia', 'Viral Pneumonia', 'Normal']
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix 생성
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot 설정
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # 레이블이 잘리는 것을 방지
    plt.tight_layout()
    plt.show()

def main():
    # 디렉토리 설정
    data_directory = '/kaggle/input/pneumonia-covid19-image-dataset'
    writable_directory = '/kaggle/working/pneumonia-covid19'
    train_directory = '/kaggle/working/train'
    val_directory = '/kaggle/working/val'
    
    # 초기 디렉토리 설정
    print("Setting up directories...")
    setup_directories(data_directory, writable_directory, train_directory, val_directory)
    
    # 데이터 분할
    print("\nSplitting data into train and validation sets...")
    prepare_data_splits(writable_directory, train_directory, val_directory)
    
    # 특징 추출기 생성
    print("\nCreating feature extractor...")
    feature_extractor = create_feature_extractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 특징 추출
    print("\nExtracting features from training data...")
    X_train, y_train = extract_features(train_directory, feature_extractor, device)
    print("\nExtracting features from validation data...")
    X_val, y_val = extract_features(val_directory, feature_extractor, device)
    
    # ADASYN 적용
    print("\nApplying ADASYN sampling...")
    X_resampled, y_resampled = apply_sampling_strategies(X_train, y_train)
    print("Original shape:", X_train.shape)
    print("Resampled shape:", X_resampled.shape)
    
    # 데이터셋 생성
    train_dataset = FeatureDataset(X_resampled, y_resampled)
    val_dataset = FeatureDataset(X_val, y_val)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 모델 생성
    num_classes = len(set(y_train))
    input_features = X_resampled.shape[1]  # 1536
    model = PneumoniaModelModified(
        num_classes=num_classes,
        input_features=input_features
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 모델 학습
    print("\nStarting model training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # 모델 평가
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device=device)

if __name__ == "__main__":
    main()