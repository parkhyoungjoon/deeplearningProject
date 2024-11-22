# ------------------------------------------------------------------------------------------
# Flask Framework에서 '/' URL에 대한 라우팅
# - 파일명 : main_views.py
# ------------------------------------------------------------------------------------------
# 모듈 로딩 ---------------------------------------------------------------------------------
from flask import Blueprint, render_template, request, redirect, url_for, current_app
from DBWEB.models.models import AgeRating
from DBWEB.lib import mylib
import pandas as pd
import string
import torch
import jsonify
import os
import torch.nn as nn
from konlpy.tag import Okt
from DBWEB import AMD
from DBWEB import DMD
from DBWEB import WMD
from DBWEB import MMD
import joblib

# Blueprint 인스턴스 생성 -------------------------------------------------------------------
main_bp = Blueprint('ageRating',__name__, url_prefix='/ageRating',template_folder='template')

# 라우팅 기능 함수 정의 ----------------------------------------------------------------------

@main_bp.route('/ratingEval/<string:category>', methods=['GET','POST'])
def ageRatingModelLearn(category):
    if request.method == 'POST':
        MODEL_PATH = os.path.join(current_app.root_path, 'static', 'models', category+'_model.pth')
        if not category =='movie': VOCAB_PATH = os.path.join(current_app.root_path, 'static', 'models', category+'_vocab.pkl')
        STOPWORDS_PATH = os.path.join(current_app.root_path, 'static', 'data', category+'_stopwords.txt')
        if not category =='movie': vocab = mylib.load_vocab(VOCAB_PATH)
        stopwords = mylib.load_stopwords(STOPWORDS_PATH)
        story = request.form.get('acontent')  # 텍스트 영역에서 값 가져오기
        if category == 'drama':
            DMD.eval()
            DMD.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
            result = mylib.detectSentiment(story,DMD,vocab,stopwords)
        elif category == 'movie':
            VEC = os.path.join(current_app.root_path, 'static', 'models', 'tfidfvectorizer.pkl')
            with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                stopwords = [i.strip() for i in f.readlines()]

            token = Okt().morphs(story)
            clean_token = mylib.remove_punctuation(token)
            clean_token = mylib.remove_stopwords(clean_token, stopwords)
            clean_text = ' '.join(clean_token)
            textDF = pd.DataFrame({'clean_text' : [clean_text]})

            loaded_vectorizer = joblib.load(VEC)
            text_vector = loaded_vectorizer.transform(textDF['clean_text']).toarray()
            text_vectorDF = pd.DataFrame(text_vector)

            MMD.eval()
            checkpoint = torch.load(MODEL_PATH)
            MMD.load_state_dict(checkpoint['model_state_dict'])

            tensor = torch.tensor(text_vectorDF.values, dtype=torch.float)
            # 출력층에서 로짓 값을 반환하므로, sigmoid로 확률을 압축
            predicted_value = torch.sigmoid(MMD(tensor.unsqueeze(1)))
            predicted_value = (predicted_value > 0.5).float().item()  # 0.5를 기준으로 0 또는 1로 변환

            if predicted_value == 0:
                predicted_value = '15세미만관람가'
            elif predicted_value == 1:
                predicted_value = '15세이상관람가'

            result = predicted_value
        elif category == 'webtoon':
            tokenizer = Okt()
            token_to_id = {token: idx for idx, token in enumerate(vocab)}
            new_reviews = [story]
            new_reviews = [mylib.re_text(review) for review in new_reviews]
            new_tokens = [[token for token in tokenizer.nouns(review) if token not in stopwords] for review in new_reviews]
            new_ids = [[token_to_id.get(token, 1) for token in tokens] for tokens in new_tokens]
            new_ids_padded = mylib.webtoon_pad_sequences(new_ids, 30, 0)
            new_ids_tensor = torch.tensor(new_ids_padded)
            ## 예측
            WMD.eval()
            with torch.no_grad():
                outputs = WMD(new_ids_tensor)
                predictions = torch.sigmoid(outputs).item()
            prediction = 1 if predictions >= 0.5 else 0
            result = '나이 제한 필요' if prediction == 1 else '전체 이용가'
        elif category == 'anime':
            tokenizer = Okt()
            AMD.eval()
            AMD.load_state_dict(torch.load(MODEL_PATH))
            punc = string.punctuation
            MAX_LENGTH = 60
            
            # 리뷰 처리
            pre_sentences = mylib.preprocess_text(story, punc)
            sentence_tokens = mylib.tokenize_and_remove_stopwords(tokenizer, [pre_sentences], stopwords)
            sentence_ids = mylib.encoding_ids(vocab, sentence_tokens, vocab.get('<unk>'))

            # 입력 데이터 패딩
            input_data = mylib.pad_sequences(sentence_ids, MAX_LENGTH, vocab.get('<pad>'))
            sentence_tensor = torch.tensor(input_data, dtype=torch.long)    
        
            # 분석
            logits = AMD(sentence_tensor)
            classesd = torch.argmax(logits, dim=1)
            classesd = classesd.item()
            ac = AgeRating.query.get(classesd)
            # acontent를 처리하는 로직을 여기에 추가할 수 있습니다.
            result = ac.age_name
        # POST 요청 후 다른 URL로 리디렉션
        return redirect(url_for('ageRating.ageRatingModelLearn', category=category, acontent=story, result=result))
    else:
        acontent = request.args.get('acontent', default='', type=str)
        result = request.args.get('result', default='', type=str)
        return render_template('agerating/agerating_form.html',category=category, acontent=acontent, result=result)