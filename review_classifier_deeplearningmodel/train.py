import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch import optim
from torch import nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import os
import string
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

# 모델 정의
class ReviewClassifierModel(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_classes,
                 n_layers, dropout=0.3, bidirectional=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.multi_classifier = nn.Linear(lstm_output_dim, n_classes)
        self.binary_classifier = nn.Linear(lstm_output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        lstm_output, _ = self.lstm(embeddings)
        last_output = lstm_output[:, -1, :]  # 마지막 타임스텝의 출력
        last_output = self.dropout(last_output)
        classesd = self.multi_classifier(last_output)
        logits = self.binary_classifier(last_output)
        return classesd, logits

# 데이터 로드 함수
def load_data(train_path, test_path):
    trainDF = pd.read_csv(train_path, usecols=[1, 2, 4])
    testDF = pd.read_csv(test_path, usecols=[1, 2, 4])
    return trainDF, testDF

# 데이터 인코딩 함수
def data_encoding(DF,labelCD):
    DF['Aspect'] = DF['Aspect'].map(lambda x: labelCD.index(x))
    DF = DF.dropna()
    DF.loc[DF['SentimentPolarity'] == -1, 'SentimentPolarity'] = 0
    return DF

# 전처리 함수
def preprocess_text(text, punc):
    # 구두점 제거
    for p in punc:
        text = text.replace(p, '')
    # 한글과 공백만 남기기
    text = re.sub('[^ ㄱ-ㅣ가-힣]+', ' ', text)
    return text

# 토큰화 및 불용어 제거 함수
def tokenize_and_remove_stopwords(tokenizer, texts, stop_words):
    tokens = [tokenizer.morphs(text) for text in texts]
    tokens = [[token for token in doc if token not in stop_words] for doc in tokens]
    return tokens

# 단어사전 구축 함수
def build_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens.copy()
    vocab += [token for token, _ in counter.most_common(n_vocab)]
    return vocab

# 인코딩 함수
def encoding_ids(token_to_id, tokens, unk_id):
    return [[token_to_id.get(token, unk_id) for token in doc] for doc in tokens]

# 패딩 함수
def pad_sequences(sequences, max_length, pad_value):
    result = []
    for seq in sequences:
        seq = seq[:max_length]
        pad_len = max_length - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        result.append(padded_seq)
    return np.array(result)

def save_vocab(vocab,save_file):
    # vocab 저장
    with open(save_file, 'wb') as f:
        pickle.dump(vocab, f)

# 클래스 가중치 계산 함수
def calculate_class_weights(train_labels, num_classes, device):
    class_counts = np.bincount(train_labels.astype('int'), minlength=num_classes)
    total_count = len(train_labels)
    class_weights = total_count / (num_classes * class_counts)
    # 텐서로 변환
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights

# 학습 함수
def model_train(model, train_loader, cl_criterion, bn_criterion, optimizer, device, interval):
    model.train()
    losses = []
    accuracies = []

    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        cl_labels = labels[:, 0].to(device)
        bn_labels = labels[:, 1].to(device).float()

        # 모델 예측
        classesd, logits = model(input_ids)

        # 손실 계산
        loss_cl = cl_criterion(classesd, cl_labels)         # 다중 분류 손실
        loss_bn = bn_criterion(logits.squeeze(), bn_labels) # 이진 분류 손실
        loss = loss_cl + loss_bn
        losses.append(loss.item())

        # 정확도 계산
        predictions = torch.argmax(classesd, dim=1)  # CrossEntropyLoss 사용 시 softmax 불필요
        accuracy = (predictions == cl_labels).float().mean().item()
        accuracies.append(accuracy)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0 and step != 0:
            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            print(f'Train Step {step}, Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}')
            losses = []
            accuracies = []

# 테스트 함수
def model_test(model, test_loader, cl_criterion, bn_criterion, device):
    model.eval()
    losses = []
    cl_true = []
    cl_pred = []
    bn_true = []
    bn_pred = []

    with torch.no_grad():
        for step, (input_ids, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            cl_labels = labels[:, 0].to(device).long()
            bn_labels = labels[:, 1].to(device).float()

            # 모델 예측
            classesd, logits = model(input_ids)

            # 손실 계산
            loss_cl = cl_criterion(classesd, cl_labels)
            loss_bn = bn_criterion(logits.squeeze(), bn_labels)
            loss = loss_cl + loss_bn
            losses.append(loss.item())

            # 다중 분류 예측
            cl_predictions = torch.argmax(classesd, dim=1)
            cl_true.extend(cl_labels.cpu().numpy())
            cl_pred.extend(cl_predictions.cpu().numpy())

            # 이진 분류 예측
            bn_predictions = (torch.sigmoid(logits) > 0.5).int().squeeze()
            bn_true.extend(bn_labels.cpu().numpy())
            bn_pred.extend(bn_predictions.cpu().numpy())

    # 손실 평균 계산
    avg_loss = np.mean(losses)

    # F1 스코어 계산
    cl_f1 = f1_score(cl_true, cl_pred, average='weighted')
    bn_accuracy = np.mean(np.array(bn_pred) == np.array(bn_true))

    # # 혼동 행렬 시각화
    # cm = confusion_matrix(cl_true, cl_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()

    # 클래스 예측 분포 확인
    unique, counts = np.unique(cl_pred, return_counts=True)
    print("Class predictions distribution:", dict(zip(unique, counts)))

    # F1 스코어와 정확도 출력
    print(f'Val Loss: {avg_loss:.4f}')
    print(f'Class F1 Score: {cl_f1:.4f}')
    print(f'Binary Accuracy: {bn_accuracy:.4f}')

    # 상세 보고서 출력
    print("\nClass Classification Report:\n", classification_report(cl_true, cl_pred))
    print("\nBinary Classification Report:\n", classification_report(bn_true, bn_pred))

    return cl_f1, avg_loss

def changeDF(DF):
    DF.loc[DF['Aspect'] == "배터리",'Aspect'] = "전력 및 품질 관련"
    DF.loc[DF['Aspect'] == "소비전력",'Aspect'] = "전력 및 품질 관련"
    return DF

# 메인 함수
def main():
    # 하이퍼파라미터 설정
    N_VOCAB = 5000
    MAX_LENGTH = 22
    EPOCHS = 500
    INTERVAL = 500
    BATCH_SIZE = 32
    LR = 0.001
    special_tokens = ['<pad>', '<unk>']

    
    # 데이터 로드
    DATA_PATH = '../DATA/'
    TRAIN_PATH = r'C:\Users\KDP-38\Documents\EX_TEXT_38\project\DATA\IT_review.csv'
    TEST_PATH = r'C:\Users\KDP-38\Documents\EX_TEXT_38\project\DATA\IT_review_test.csv'
    print("Training data path:", TRAIN_PATH)
    trainDF, testDF = load_data(TRAIN_PATH, TEST_PATH)
    trainDF, testDF = changeDF(trainDF), changeDF(testDF)
    

    aspectCD = trainDF.Aspect.unique().tolist()
    # 데이터 인코딩
    trainDF = data_encoding(trainDF,aspectCD)
    testDF = data_encoding(testDF,aspectCD)

    # 전처리: 구두점 제거 및 특수문자 제거
    punc = string.punctuation
    trainDF['SentimentText'] = trainDF['SentimentText'].apply(lambda x: preprocess_text(x, punc))
    testDF['SentimentText'] = testDF['SentimentText'].apply(lambda x: preprocess_text(x, punc))
    
    # 불용어 로드
    STOP_WORD =  r'C:\Users\KDP-38\Documents\EX_TEXT_38\project\DATA\stopwords.txt'
    with open(STOP_WORD, 'r', encoding='utf-8') as f:
        stop_words = set([line.strip() for line in f])

    # 토큰화 및 불용어 제거
    tokenizer = Okt()
    train_tokens = tokenize_and_remove_stopwords(tokenizer, trainDF['SentimentText'], stop_words)
    test_tokens = tokenize_and_remove_stopwords(tokenizer, testDF['SentimentText'], stop_words)
    
    # 단어 사전 구축
    vocab = build_vocab(train_tokens, N_VOCAB, special_tokens)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}

    # 인코딩 및 패딩
    pad_id = token_to_id['<pad>']
    unk_id = token_to_id['<unk>']
    train_ids = encoding_ids(token_to_id, train_tokens, unk_id)
    test_ids = encoding_ids(token_to_id, test_tokens, unk_id)
    train_ids = pad_sequences(train_ids, MAX_LENGTH, pad_id)
    test_ids = pad_sequences(test_ids, MAX_LENGTH, pad_id)

    # 텐서화
    train_ids = torch.tensor(train_ids, dtype=torch.long)
    test_ids = torch.tensor(test_ids, dtype=torch.long)

    # 레이블 텐서화
    train_labels = torch.tensor(list(zip(trainDF['Aspect'].values, trainDF['SentimentPolarity'].values)), dtype=torch.long)
    test_labels = torch.tensor(list(zip(testDF['Aspect'].values, testDF['SentimentPolarity'].values)), dtype=torch.float32)

    # 데이터셋 생성
    train_dataset = TensorDataset(train_ids, train_labels)
    test_dataset = TensorDataset(test_ids, test_labels)

    # 클래스 가중치 계산
    train_labels_np = trainDF['Aspect'].values
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = calculate_class_weights(train_labels_np, len(aspectCD), device)
    print("Scaled Class Weights:", class_weights)

    # 샘플 가중치 계산 및 WeightedRandomSampler 생성
    sample_weights = class_weights[train_labels_np].cpu()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 모델 초기화
    n_vocab = len(token_to_id)
    hidden_dim = 64
    embedding_dim = 128
    n_layers = 2
    classifier = ReviewClassifierModel(
        n_vocab=n_vocab,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_classes=len(aspectCD),
        n_layers=n_layers,
        dropout=0.3
    ).to(device)

    # 손실 함수 및 옵티마이저 설정
    cl_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    bn_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 조기 종료 및 모델 저장을 위한 변수 초기화
    best_f1 = 0.0
    epochs_no_improve = 0
    n_epochs_stop = 5  # 조기 종료를 위한 patience

    # 모델 저장 디렉토리 생성
    SAVE_PATH = r'C:\Users\KDP-38\Documents\EX_TEXT_38\project\models'
    SAVE_VOCAB = os.path.join(SAVE_PATH, 'it_vocab.pkl')
    SAVE_FILE = os.path.join(SAVE_PATH, 'it_model.pth')
    SAVE_MODEL = os.path.join(SAVE_PATH, 'it_model_full.pth')
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    save_vocab(vocab,SAVE_VOCAB)
    f1_history = []
    loss_history = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        model_train(classifier, train_loader, cl_criterion, bn_criterion, optimizer, device, INTERVAL)
        current_f1, current_loss = model_test(classifier, test_loader, cl_criterion, bn_criterion, device)
        f1_history.append(current_f1)
        loss_history.append(current_loss)
        # 학습률 스케줄러 업데이트
        scheduler.step(current_loss)
        
        # F1 스코어가 개선되었는지 확인
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_no_improve = 0
            # 모델 저장
            torch.save(classifier.state_dict(), SAVE_FILE)
            torch.save(classifier, SAVE_MODEL)
            print(f'F1 Score improved to {current_f1:.4f}. Model saved.')
        else:
            epochs_no_improve += 1
            print(f'No improvement in F1 Score for {epochs_no_improve} epochs.')
        
        # 조기 종료 조건
        if epochs_no_improve >= n_epochs_stop:
            print("Early stopping triggered.")
            break
    SAVEDF = pd.DataFrame({'f1_score':f1_history, 'loss':loss_history})
    SAVEDF.to_csv(r'C:\Users\KDP-38\Documents\EX_TEXT_38\project\DATA\score_loss_history.csv')
if __name__ == "__main__":
    main()
