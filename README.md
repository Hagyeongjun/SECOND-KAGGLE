import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


arr_1 = np.load('train.npy', allow_pickle=True)
arr_2 = np.load('test.npy', allow_pickle=True)

dic = arr_1.tolist()
test = arr_2.tolist()

x = dic["input"].reshape(4608, 22, 1125)
y = dic["label"]
test = test["input"].reshape(576, 22, 1125)

lb = LabelBinarizer()
y = lb.fit_transform(y)
y = np.argmax(y, axis=1)

# 채널마다 표준화 사용
scalers = {}
x_scaled = np.zeros_like(x)
test_scaled = np.zeros_like(test)

for i in range(22):
    scalers[i] = StandardScaler()
    x_channel = x[:, i, :]
    test_channel = test[:, i, :]

    x_scaled[:, i, :] = scalers[i].fit_transform(x_channel)
    test_scaled[:, i, :] = scalers[i].transform(test_channel)

x = x_scaled
test = test_scaled


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=777, stratify=y)
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()
test_real = torch.tensor(test).float()

class ConvNet(nn.Module):
    def __init__(self, num_classes, dropout):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU())
        self.drop_out = nn.Dropout(dropout)
        self.fc1 = nn.Linear(17920, 500)
        self.drop_out = nn.Dropout(dropout)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

num_classes = 4
learning_rate = 0.001
weight_decay = 0.003
batch_size = 50
drop_out = 0.5
num_epochs = 10
k_folds=5
best_diff = np.inf

net = ConvNet(num_classes=num_classes, dropout=drop_out).to(device)
loss_fn = nn.NLLLoss()
optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

# 모델 트레인 함수
def trainModel(net, optimizer, scheduler, num_epochs):
    global best_diff
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=777)
    trainAcc = []
    testAcc = []
    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        fold_train_losses = []
        fold_val_losses = []

        for fold, (train_idx, val_idx) in enumerate(k_fold.split(X_train, y_train)):
            print(f'Fold {fold + 1}/{k_folds}')

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            net.train()
            running_loss = 0.0
            for i in range(0, len(X_fold_train), batch_size):
                inputs = X_fold_train[i:i + batch_size].unsqueeze(1).to(device)
                labels = y_fold_train[i:i + batch_size].to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / (len(X_fold_train) / batch_size)
            fold_train_losses.append(train_loss)

            val_loss = testModelLoss(net, X_fold_val, y_fold_val)
            fold_val_losses.append(val_loss)

            print(f'Fold {fold + 1} Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        avg_train_loss = np.mean(fold_train_losses)
        avg_val_loss = np.mean(fold_val_losses)
        avg_test_loss = testModelLoss(net, X_test, y_test)
        diff = abs(avg_val_loss - avg_train_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Diff: {diff:.4f}')

        trainAcc.append(testModel(net, X_train, y_train))
        testAcc.append(testModel(net, X_test, y_test))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        test_losses.append(avg_test_loss)

        scheduler.step()

        if diff < best_diff:
            best_diff = diff
            torch.save(net.state_dict(), 'best_model.pth')
            print("최적의 모델 저장")

    return trainAcc, testAcc, train_losses, val_losses, test_losses


# 정확도 계산 함수
def testModel(net, X_test, Y_test):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size].unsqueeze(1).to(device)
            labels = Y_test[i:i + batch_size].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


# 모델 저장 위한 loss계산 함수
def testModelLoss(net, X_test, Y_test):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size].unsqueeze(1).to(device)
            labels = Y_test[i:i + batch_size].to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / (len(X_test) / batch_size)


# 혼동 행렬 생성 함수
def generateConfusionMatrix(net, X_test, Y_test):
    preds = np.array([], dtype=int)
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size].unsqueeze(1).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            preds = np.concatenate((preds, predicted.cpu().numpy()))
    return preds, confusion_matrix(Y_test.cpu().numpy(), preds, normalize='true')


#metric 계산
def metric(net, X_test, Y_test):
    preds, conf_matrix = generateConfusionMatrix(net, X_test, Y_test)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_test.cpu().numpy(), preds, average='micro')

    specificity_per_class = []
    for i in range(num_classes):
        tn = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tn

        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    avg_specificity = np.mean(specificity_per_class)

    return precision, recall, f1, avg_specificity


# 혼동행렬 그리는 함수
def ConfusionMatrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# ROC 곡선과 AUC 점수 함수
def ROCCurve(net, X_test, y_test, num_classes):
    y_test_bin = LabelBinarizer().fit_transform(y_test.cpu().numpy())
    y_score = np.zeros((y_test_bin.shape[0], num_classes))

    net.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i + batch_size].unsqueeze(1).to(device)
            outputs = net(inputs).cpu().numpy()
            y_score[i:i + batch_size] = outputs

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f'ROC curve (area = {roc_auc["micro"]:.2f})', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()

    # Print AUC scores
    print(f'Micro-average AUC: {roc_auc["micro"]:.4f}')


# t-SNE 시각화 함수
def Tsne(net, X_test, y_test):
    net.eval()
    features = []
    labels = y_test.cpu().numpy()

    with torch.no_grad():
        inputs = X_test.unsqueeze(1).to(device)
        outputs = net.layer5(net.layer4(net.layer3(net.layer2(net.layer1(inputs)))))
        outputs = outputs.view(outputs.size(0), -1)
        features = outputs.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=777)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        indices = np.where(labels == i)
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {i}', alpha=0.5)
    plt.legend()
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Features')
    plt.show()

# train, valid, test loss 그리는 함수
def Loss(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

# 모델 학습

trainAcc, testAcc, train_losses, val_losses, test_losses = trainModel(net, optimizer, scheduler, num_epochs)

# 모델 초기화
net = ConvNet(num_classes=num_classes, dropout=0.5).to(device)

# 저장된 모델 상태 로드
net.load_state_dict(torch.load('best_model.pth'))


class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
preds, conf_matrix = generateConfusionMatrix(net, X_test, y_test)
precision, recall, f1, specificity = metric(net, X_test, y_test)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Specificity: {specificity:.4f}")

# 혼동 행렬 그리기
ConfusionMatrix(conf_matrix, class_names)

# ROC곡선 그리고 점수 계산하기
ROCCurve(net, X_test, y_test, num_classes)

# t-SNE 사용해서 시각화
Tsne(net, X_test, y_test)

# train, valid, test loss 그리기
Loss(train_losses, val_losses, test_losses)

# 진짜 test set에 대한 예측 생성
with torch.no_grad():
    test_real_inputs = test_real.unsqueeze(1).type(torch.FloatTensor).to(device)
    outputs = net(test_real_inputs)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()

# CSV 파일 만들기
df = pd.DataFrame(predicted, columns=["TARGET"])
df.to_csv('output.csv', index_label="ID")
