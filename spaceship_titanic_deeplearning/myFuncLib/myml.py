import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import myFuncLib.mygraph as myGraph
import myFuncLib.myfunc as myFunc

class myML:
    def __init__(self,model):
        self.model = model

    def train_test_split_reset_idx(featureDF, targetDF):
        X_train, X_test, Y_train, Y_test = train_test_split(featureDF,
                                                            targetDF,
                                                            stratify=targetDF,
                                                            test_size=25,
                                                            random_state=10)
        myFunc.shape_view(X_train, X_test, Y_train, Y_test)
        X_train=X_train.reset_index(drop=True)
        Y_train=Y_train.reset_index(drop=True)
        X_test=X_test.reset_index(drop=True)
        Y_test=Y_test.reset_index(drop=True)

        return X_train, X_test, Y_train, Y_test

    def train_test_scal_data(X_train,X_test):
        # 스케일러 인스턴스 생성
        mmScaler = MinMaxScaler()

        # 스케일러에 데이터셋 전용에 속성값 설정
        mmScaler.fit(X_train)
        print('[스케일러 데이터]')
        print(f'min: {mmScaler.min_}, '
            f'scale: {mmScaler.scale_}, '
            f'min: {mmScaler.data_min_}, '
            f'max: {mmScaler.data_max_}, ')
        
        # 학습용, 테스트용 데이터셋 스케일링 진행
        X_train_scaled = mmScaler.transform(X_train)
        X_test_scaled = mmScaler.transform(X_test)

        return mmScaler, X_train_scaled, X_test_scaled

    def train_test_auto_split(self,featureDF, targetDF):
        X_train, X_test, Y_train, Y_test = self.train_test_split_reset_idx(featureDF, targetDF)
        mmScaler, X_train_scaled, X_test_scaled = self.train_test_scal_data(X_train,X_test)
        return mmScaler, X_train_scaled, X_test_scaled, Y_train, Y_test

    def best_k_find(model,X_test_scaled,Y_test):
        scores = {}
        for k in range(1, model.n_samples_fit_+1):
            model.n_neighbors=k

            score=model.score(X_test_scaled,Y_test)

            scores[k]=score
        best_k=sorted(scores.items(), key=lambda x:x[1], reverse=True)[0][0]
        myGraph.draw_plot_graph(scores,'Ks','Scores',yline=[best_k])
        return best_k

    def learn_data_set(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
        print(self.model.classes_, self.model.feature_names_in_, self.model.n_samples_fit_)

    def find_k_neigh(self,new_data):
        distance, index = self.model.kneighbors(new_data)
        print(index,distance)
        neighbors = index.reshape(-1).tolist()
        return neighbors
    
    def get_mean_error_score(self, DF, SR): # DF: 예측 데이터 SR 정답
        # 성능지표 => 오차계산과 결정계수 계산
        pre_jumsu = self.model.predict(DF)

        # 손실/비용 계산 함수 ==> 정닶과 예측값 제공
        mse = mean_squared_error(SR,pre_jumsu)
        rmse = mean_squared_error(SR, pre_jumsu, squared=False)
        mae = mean_absolute_error(SR,pre_jumsu)

        # 얼마나 정답에 가깝게 값을 예측 했는냐를 나타내는 지표, ==> 정답과 예측값 제공 1에 가까울수록 좋음
        r2 = r2_score(SR, pre_jumsu)
        ## 손실/비용함수 값은 0에 가까울수록
        ## 결정계수 값은 1에 가까울 수록 성능 좋은 모델
        print(f'mse: {mse}')
        print(f'rmse: {rmse}')
        print(f'mae: {mae}')
        print(f'r2: {r2}')