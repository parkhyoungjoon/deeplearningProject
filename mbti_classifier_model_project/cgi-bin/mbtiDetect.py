# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import joblib      # AI 모델 관련
import sys, codecs # 인코딩 관련
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys, os
from models.mlpmodel import MLPModel


# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(words, msg, in_scores,red_focus,res_show):
    print("Content-Type: text/html; charset=utf-8")
    print("""
    
        <!DOCTYPE html>
        <html lang="ko">
         <head>
          <meta charset="UTF-8">
          <title>---AI언어판별---</title>
          """+
          """
          <style>
            *{
                margin: 0;
                padding:0;
            }
            body {
                background-color: #f0f0f0;
            }
            div.wrap{
                display: flex; 
                height:100vh;
                justify-content: center; 
                align-items: center;
            }
            div.wrap_inner{
                float:left;
                width:60%;
                max-width: 800px;
                margin:0 auto;
            }
            div.title{
                float:left;
                width:100%;
                text-align:center; 
                margin-bottom:30px;
                font-size:40px;
                font-weight:bold;
                color:#495057;
            }
            div.imgbox{
                width:50%;
                float:left;
            }
            div.imgbox > img {width:100%;}
            div.mbti{
                width:100%;
                float:left;
                padding:10px 0 10px 0;
                margin-top:30px;
                background-color:#495057;
                border:solid #212529 4px;
                color:#f1f3f5;
                font-size:30px;
                font-weight:bold;
                text-align:center;
            }
            div.u-list{
                float:left;
                width:46%;
                padding:0 2% 0 2%;
            }
            ul {
                width:100%;
                float:left;
                list-style-type: none;
                padding: 0; 
            }
            li {
                width:100%;
                margin: 10px 0;
            }
            input[type="text"] {
                width: 90%; 
                padding: 11px; 
                border: 2px solid #ccc; 
                border-radius: 5px; 
                font-size: 16px; 
                transition: border-color 0.3s;
            }
            input[type="text"]:hover{
                border-color: #D1180B;
            }
            input[type="submit"]{
                width: 100%;
                height : 80px;
                border: 2px solid #9f8473;
                border-radius: 5px;
                background-color: #EFE7DA;
                font-size:20px;
                font-weight:bold;
                color: #9f8473;
            }
            input[type="submit"]:hover{
                border-color: #6c5d53;
                color: #6c5d53;
            }
            div.score > ul > li {
                width:25%;
                float:left;
                color:#495057;
                font-size:17px;
                font-weight:bold;
                }
          </style>
         </head>
         <body>
          <form>
          """)
    print(f"""
            <div class='wrap'>
                <div class='wrap_inner'>
                    <div class='title'> MBTI 진단모델 </div>
                    <div class='u-list'>
                        <ul>
                            <li><input type='text' name='word1' placeholder='영어만 입력해주세요' value={words[0]}></li>
                            <li><input type='text' name='word2' placeholder='영어만 입력해주세요' value={words[1]}></li>
                            <li><input type='text' name='word3' placeholder='영어만 입력해주세요' value={words[2]}></li>
                            <li><input type='text' name='word4' placeholder='영어만 입력해주세요' value={words[3]}></li>
                            <li><input type='text' name='word5' placeholder='영어만 입력해주세요' value={words[4]}></li>
                            <li><input type="submit" value="mbti 진단"></li>
                        </ul>
                    </div>
                    <div class='imgbox'><img src='https://blog.kakaocdn.net/dn/o6VXe/btsINc3wa3a/kf57MpXyny0cr6m9TCj2Gk/img.png'/></div>
                    <div class='mbti'>{msg}</div>
    """)
    if(res_show):
        print(f"""
                    <div class='score'>
                        <ul>
                            <li {red_fos(red_focus[0])}>E일확률: {in_scores[0]}%</li>
                            <li {red_fos(red_focus[1])}>S일확률: {in_scores[1]}%</li>
                            <li {red_fos(red_focus[2])}>F일확률: {in_scores[2]}%</li>
                            <li {red_fos(red_focus[3])}>P일확률: {in_scores[3]}%</li>
                            <li {red_fos(1-red_focus[0])}>I일확률: {round(100-in_scores[0],2)}%</li>
                            <li {red_fos(1-red_focus[1])}>N일확률: {round(100-in_scores[1],2)}%</li>
                            <li {red_fos(1-red_focus[2])}>T일확률: {round(100-in_scores[2],2)}%</li>
                            <li {red_fos(1-red_focus[3])}>J일확률: {round(100-in_scores[3],2)}%</li>
                        </ul>
                    </div>
                </div>
            </div>
          </form>
         </body>
        </html>""")

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detectLang
# 재 료 : 사용자 입력 데이터
# 결 과 : 판별 언어명(영어, 프랑스~)

def red_fos(focus):
    if focus:
        return "style='color:#D1180B;'"
    else:
        return ''

def wordtoVector(text,loaded_vectorizer):
    return loaded_vectorizer.transform(text).toarray()

def detectMbti(resultWord, snmodel):
    score = 0
    perce = 0
    for word in resultWord:
        dataTS = torch.FloatTensor(word)
        # 새로운 데이터에 대한 예측 즉, predict
        snmodel.eval()
        with torch.no_grad():
        #     # 추론 / 평가
            outputs = snmodel(dataTS).view(-1)
            predicted = torch.round(torch.sigmoid(outputs)).item()
            perced = torch.sigmoid(outputs).item() * 100
            score += int(predicted)
            perce += float(perced)
    return checkMbti(score,perce)

def checkMbti(score,perce):
    r_score = score / 5
    r_perce = round(perce / 5,2)
    if round(r_score): return 1, r_perce
    else : return 0, r_perce

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정

model_names = ['ei_model','sn_model','ft_model','pj_model']
model_paths = []
models = []
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    for mod in model_names:
        model_paths.append(os.path.dirname(__file__)+ '/models/'+mod+'.pth') # 웹상에서는 절대경로만
else:
    for mod in model_names:
        model_paths.append('./models/'+mod+'.pth') 
        
    
VECTOR_LINK = os.path.dirname(__file__)+ '/models/tfidf_vectorizer.pkl'
loaded_vectorizer = joblib.load(VECTOR_LINK)

for pat in model_paths:
    models.append(torch.load(pat, weights_only=False,map_location=torch.device('cpu')))

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태크 속 데이터 가져오기
words = []
words.append(form.getvalue("word1", default="").lower())
words.append(form.getvalue("word2", default="").lower())
words.append(form.getvalue("word3", default="").lower())
words.append(form.getvalue("word4", default="").lower())
words.append(form.getvalue("word5", default="").lower())
#text ="Happy New Year" # 테스트용 (쥬피터 내부)

mbti_list = [['I','E'],['N','S'],['T','F'],['J','P']]
# (3-3) 판별하기
msg =""
if not '' in words:
    scores = []
    red_focus = []
    res_show = True
    resultWord = wordtoVector(words,loaded_vectorizer)
    for i in range(len(models)):
        mbti, score = detectMbti(resultWord,models[i])
        msg += mbti_list[i][mbti]
        scores.append(score) # 1일확률
        red_focus.append(mbti) 
else:
    scores = [0,0,0,0]
    red_focus = [0,0,0,0]
    res_show = False
    msg += 'mbti 결과'
        
# (4) 사용자에게 WEB 화면 제공
showHTML(words,msg,scores,red_focus,res_show)