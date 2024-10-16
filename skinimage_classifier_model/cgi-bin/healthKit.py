# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import io, cgi, cgitb  # cgi 프로그래밍 관련
import sys, codecs # 인코딩 관련
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys, os
from torchvision.transforms import v2
import torchvision.transforms as transforms
from models.skinkitmodel import SkinKitModel
import CNNutils as utils
import torchvision.models as models
# from models.mlpmodel import MLPModel
from PIL import Image

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(val):
    print("Content-Type: text/html; charset=utf-8")
    print("""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Upload</title>
        <style>
            body {
                font-family: 'Malgun gothic';
                background-color: #96C8A2;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }

            .upload-container {
                float:left;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                height:500px;
                text-align: center;
                width: 100%;
            }

            .upload-container h2 {
                color: #333;
                margin-bottom: 20px;
            }

            .file-input {
                display: none;
            }

            .custom-file-upload {
                display: inline-block;
                padding: 12px 25px;
                cursor: pointer;
                background-color: #007bff;
                color: #fff;
                border-radius: 5px;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }

            .custom-file-upload:hover {
                background-color: #0056b3;
            }

            .image-view {
                margin-top: 20px;
                max-width: 250px;
                max-height:250px;
                border: 1px solid #ddd;
                border-radius: 5px;
                display: none;
            }

            .submit-btn {
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #28a745;
                color: #fff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }

            .submit-btn:hover {
                background-color: #218838;
            }
            .wrap{
                width:100%;
                max-width: 800px;
                float:left;
            }
            .wrap_in{
                width:100%;
                max-width: 800px;
                float:left;
            }
            .wrap_header {
                float:left;
                width:100%;
            }
            .wrap_body{
                float:left;
                margin-top:10px;
                width:100%;
            }
            .logo-container {
                width:100%;
                height:100px;
                display: flex;
                justify-content: center;
                align-items: center;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .safety {
                font-size: 34px;
                font-weight: bold;
                color: #2b7a78;
                letter-spacing: 2px;
            }

            .kit {
                font-size: 40px;
                color: #eeeeee;
                font-weight: bold;
                letter-spacing: 1px;
            }
            select {
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 5px;
                width: 200px;
                background-color: #fff;
            }

            select:focus {
                outline: none;
                border-color: #3498db;
            }
            .imgGallery{
                height:250px;
                width:50%;
                float:left;
            }
            ul li {list-style:none;}
            ul>li:first-child{
                font-size:20px;
                font-weight:bold;
            }
        </style>
    </head>
    """)
    print(f"""
    <body>
        <div class="wrap">
            <div class="wrap_in">
                <div class="wrap_header">
                    <div class="logo-container">
                        <span class="safety">Safety <span class="kit">Kit</span></span>
                    </div>
                </div>
                <div class="wrap_body">
                    <div class="upload-container">
                        <h2>이미지를 업로드해주세요</h2>
                        <form id="upload-form" action="" method="POST" enctype="multipart/form-data" onsubmit="alert_click(event)">
                            <select name="imageSelector" id="imageSelector" onchange="exampleImage()">
                                <option value="">모델을 고르세요</option>
                                <option value="0">피부모델</option>
                                <option value="1">폐 모델</option>
                                <option value="2">뇌 모델</option>
                            </select>
                            <label for="file-upload" class="custom-file-upload">
                                이미지 선택
                            </label>
                            <input id="file-upload" class="file-input" type="file" accept="image/*" name="image" onchange="previewImage(event)">
                            <button type="submit" class="submit-btn">Upload</button>
                            <div>{val}</div>
                            <div class="imgGallery">
                                <ul>
                                <li>Image Preivew</li>
                                <li><img id="image-preview" class="image-view" alt="Image Preview"></li>
                                </ul>
                            </div>
                            <div class="imgGallery">
                                <ul>
                                    <li>Image Example</li>
                                    <li><img id="image-example" class="image-view" alt="Image example"></li>
                                </ul>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    """)
    print("""
        <script>
            var img_ex = ['https://i.postimg.cc/7bhs6Xr7/skin-example.jpg','https://i.postimg.cc/5jrPFvfr/lung-example.png','https://i.postimg.cc/GT4q53xz/brain-example.jpg']; 
            
            function alert_click(event){
                var imageSelect = document.getElementById("imageSelector");
                var selectedImage = imageSelect.value;
          
                if (selectedImage === ''){
                    alert('모델을 선택해주세요!')
                    event.preventDefault();
                }
            }
          
            function previewImage(event) {
                const preview = document.getElementById('image-preview');
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                }
            }
            function exampleImage(event) {
                
                var imageSelect = document.getElementById("imageSelector");
                var selectedImage = imageSelect.value;

                // 이미지 디스플레이 영역 업데이트
                var imageDisplay = document.getElementById("image-example");
                if (selectedImage === ''){
                    imageDisplay.style.display = 'None';
                }else{
                    imageDisplay.src = img_ex[selectedImage];
                    imageDisplay.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    """)

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detectLang
# 재 료 : 사용자 입력 데이터
# 결 과 : 판별 언어명(영어, 프랑스~)

def image_chage(img,img_attr):
    transConvert = v2.Compose(img_attr)
    img = transConvert(img)
    return img

def red_fos(focus):
    if focus:
        return "style='color:#D1180B;'"
    else:
        return ''
    
def get_disease(num):
    if num==0:
        disease='코로나'
    elif num==1:
        disease='정상'
    elif num==2:
        disease='폐렴'
    elif num==3:
        disease='결핵'
    return disease

def get_skin_di(num):
    # return num
    target = pd.read_csv(os.path.dirname(__file__)+'/csv/skin_diseases.csv',header = None)
    return target.loc[num,target.columns[1]]

def get_brain_di(num):
    return '뇌졸중 가능성있음' if (num>0.5).item() else '뇌졸중 확률이 낮음'



# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정

model_names = ['skin_model','lung_model','brain_model']
model_list = []
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    for mod in model_names:
        model_path = os.path.dirname(__file__)+ '/models/'+mod+'.pth'  # 웹상에서는 절대경로만
        model_list.append(torch.load(model_path, weights_only=False,map_location=torch.device('cpu')))

    
# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()
image_attr = [[
        v2.ToImage(),
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True)
    ],[
        v2.ToImage(),
        v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True)
    ],[
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]]

# (3-2) Form안에 image 태크 속 데이터 가져오기
img_file = form.getvalue('image', default='')
imgs = form.getvalue('imageSelector', default='')
if imgs == 2 :
    model=models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc=nn.Sequential(
                nn.Linear(512,1024),
                nn.Linear(1024,512),
                nn.Linear(512,1),
    )
    for param in model.fc.parameters():
        param.requires_grad=True
if imgs != '' : imgs = int(imgs)
skin_dis_list = pd.read_csv(os.path.dirname(__file__)+ '/csv/skin_diseases.csv')
# (3-3) 판별하기
msg =""
val=''
if img_file != '':
    # img_bytes = img_file.file.read()  # bytes 형식으로 읽어옵니다.
    img_file = Image.open(io.BytesIO(img_file))
    img = image_chage(img_file,image_attr[imgs])
    img = img.unsqueeze(dim=0).float()
    pre = model_list[imgs](img)
    if (imgs == 0):
        pre_max = int(torch.max(torch.softmax(pre,dim=1),dim=1).indices)
        val = get_skin_di(pre_max)
    elif (imgs == 1):
        pre_max = torch.argmax(pre) 
        val = get_disease(pre_max)
    else:
        val = get_brain_di(pre)
else:
    val=''
        
# (4) 사용자에게 WEB 화면 제공
showHTML(val)