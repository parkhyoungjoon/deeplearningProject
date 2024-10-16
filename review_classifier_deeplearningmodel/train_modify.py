import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from Modules.reviewclassifiermodel import reviewClassifierModel
from torch import nn

def load_data(csvfile):                                          # 로드 데이터
    reviewDF = pd.read_csv(csvfile,usecols=[1,2,4])  # CSV => DataFrame

    trainDF = reviewDF.groupby('Aspect').apply(lambda x: x.sample(frac=0.8)).reset_index(drop=True)
    testDF = reviewDF.drop(index=trainDF.index)
    # print(trainDF.shape, trainDF.ndim)
    # print(testDF.shape, testDF.ndim)
    return trainDF, testDF

def data_encoding(DF):
    labelCD = DF.Aspect.unique().tolist()
    # print(labelCD)
    DF['Aspect'] = DF['Aspect'].map(lambda x:labelCD.index(x))
    DF.loc[DF['SentimentPolarity'] == -1,'SentimentPolarity'] = 0

    return DF, labelCD
def buid_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()                                     # count 인스턴스 생성
    for tokens in corpus:
        counter.update(tokens)                              # 카운트 인스턴스로 빈도 확인
    vocab = special_tokens                                  # vocab에서 special_tokens 추가
    for token, count in counter.most_common(n_vocab):       # counter 안의 데이터 주어진 개수 만큼 vocab에 추가
        vocab.append(token)
    return vocab                                            # 단어 사전 리턴

def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:                                # 토큰 패딩 시작
        sequence = sequence[:max_length]                      # max_length 에서 끊기
        pad_length = max_length - len(sequence)               # max_length에서 토큰 개수 빼기 == 토큰개수가 문자열보다 적다면
        padded_sequence = sequence + [pad_value] * pad_length # 적은 막큼 pad_value append
        result.append(padded_sequence)
    return np.asarray(result)                       # array 와 asarray의 차이 array copy = True, asarray copy = False 원본이 바뀌면 asarray도 변경됨

def encoding_ids(token_to_id, tokens, unk_id):
    return [
        [token_to_id.get(token, unk_id) for token in review] for review in tokens         # train data 토큰들을 사전으로 인코딩
    ]
def model_train(model, datasets, cl_criterion,bn_criterion, optimizer, device, interval):
    model.train()
    losses = list()

    for step, [input_ids, labels] in enumerate(datasets):
        input_ids = input_ids.to(device)
        print(labels.dtype)
        cl_labels = labels[:,0]
        bn_label = labels[:,1]
        bn_label = bn_label.float()
        print(cl_labels.dtype)
        print(cl_labels)
        print(bn_label.dtype)
        print(bn_label)
        classesd, logits = model(input_ids)
        print(classesd.dtype)
        print(classesd)
        print(logits.dtype)
        print(logits)
        loss = cl_criterion(classesd, cl_labels).item()
        loss += bn_criterion(logits, bn_label.view(-1,1)).item()
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % interval == 0:
            print(f'Train Loss {step} : {np.mean(losses)}')

def model_test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()
    
    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        yhat = torch.sigmoid(logits) > 0.5
        corrects.extend(
            torch.eq(yhat, labels).cpu().tolist()
        )
        print(f'Val Loss : {np.mean(losses)} , Val Accuracy : {np.mean(corrects)}')

def token_to_embeding(classifier,vocab):
    result_dict = dict()
    embedding_matrix = classifier.embedding.weight.detach().cpu.numpy()
    for word, emb in zip(vocab,embedding_matrix):
        token_to_embeding[word] = emb
    token = vocab[1000]
    print(token, result_dict[token])
    return result_dict

def main():
    # 변수 지정
    N_VOCAB = 5000                                          # 단어사전(vocab)에 들어갈 단어개수
    MAX_LENGTH = 32                                         # 패딩할 최대 길이
    EPOCHS = 1                                              # 에포크 반복횟수
    INTERVAL = 500                                          # 로스, 점수 확인하는 배수
    BATCH_SIZE = 16                                         # 데이터로더 배치사이즈
    LR = 0.001                                              # 데이터로더 learning rate
    special_tokens=['<pad>','<unk>']                        # specail tokens 리스트

    # 데이터 로드
    trainDF, testDF = load_data('./DATA/IT_reivew.csv')

    trainDF, aspectCD = data_encoding(trainDF)
    testDF, _ = data_encoding(testDF)
    # 데이터 전처리
    ## 형태소 추출 토큰화
    tokenizer = Okt()                                       # Okt 인스턴스 생성
    train_tokens = [tokenizer.morphs(review) for review in trainDF['SentimentText']]      # train data text의 형태소 추출
    test_tokens = [tokenizer.morphs(review) for review in testDF['SentimentText']]
    # print(trainDF.columns)

    vocab = buid_vocab(train_tokens, N_VOCAB, special_tokens)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    ## 패딩
    pad_id = token_to_id['<pad>']                                           # padding 할 pad id
    unk_id = token_to_id['<unk>']                                           # 사전에 없는 단어에 넣을 unk id
    train_ids = encoding_ids(token_to_id,train_tokens, unk_id)
    test_ids = encoding_ids(token_to_id,test_tokens, unk_id)
    train_ids = pad_sequences(train_ids, MAX_LENGTH, pad_id)
    test_ids = pad_sequences(test_ids, MAX_LENGTH, pad_id)

    ## 데이터 텐서화
    train_ids = torch.tensor(train_ids)
    test_ids = torch.tensor(test_ids)
    train_labels = torch.tensor(list(zip(trainDF['Aspect'].values, trainDF['SentimentPolarity'].values)), dtype=torch.long)
    test_labels = torch.tensor(list(zip(testDF['Aspect'].values, testDF['SentimentPolarity'].values)), dtype=torch.float32)

    # print(max([tensor.size(0) for tensor in train_ids]))

    ## 데이터셋 생성
    train_dataset = TensorDataset(train_ids,train_labels)
    test_dataset = TensorDataset(test_ids,test_labels)

    ## 데이터 로더 생성
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # 학습 인슨턴스 생성
    n_vocab = len(token_to_id)
    hidden_dim = 64
    embedding_dim = 128
    n_layers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = reviewClassifierModel(
        n_vocab=n_vocab, hidden_dim=hidden_dim,embedding_dim=embedding_dim,n_classes=len(aspectCD),n_layers=n_layers
    ).to(device)   # lstm 분류 모델 생성

    cl_criterion = nn.CrossEntropyLoss().to(device)
    bn_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.RMSprop(classifier.parameters(), lr=LR)
    
    # 모델 학습 
    for epoch in range(EPOCHS):
        model_train(classifier,train_loader,cl_criterion, bn_criterion, optimizer, device, INTERVAL)
        # model_test(classifier,test_loader,cl_criterion, be_criterion , device)
    
    # token_to_embeding(classifier,vocab)
main()

