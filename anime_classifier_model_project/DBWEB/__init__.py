# ----------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : __init__.py
# ----------------------------------------------------------------------------
# 모듈로딩 --------------------------------------------------------------------
# Application 생성 함수
# - 함수명 : create_app() <=== 변경 불가!
# ----------------------------------------------------------------------------
# 모듈로딩
from flask import Flask, render_template

# DB관련 설정
import config
import torch.nn.functional as F
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from DBWEB.lib.animeclassifiermodel import AnimeClassifierModel
from DBWEB.lib.sentenceclassifier import SentenceClassifier
from DBWEB.lib.textmodel import TextModel
DMD = SentenceClassifier(n_vocab=5002, hidden_dim=128, embedding_dim=128, n_layers=2, n_classes=1)
WMD = SentenceClassifier(n_vocab=10002, hidden_dim=128, embedding_dim=128, n_layers=7, n_classes=1,model_type='rnn')
AMD = AnimeClassifierModel(n_vocab=5002,hidden_dim=128,embedding_dim=128,n_classes=4,n_layers=3,dropout=0.2)
MMD = TextModel(input_size=20000,output_size = 1, hidden_list = [100],act_func=F.relu, model='lstm', num_layers=2)
DB = SQLAlchemy()
MIGRATE = Migrate()

def create_app():
    # Flask Web Server 인스턴스 생성
    APP = Flask(__name__)

    # DB 관련 초기화 설정
    APP.config.from_object(config)

    # DB 초기화 및 연동
    DB.init_app(APP)
    MIGRATE.init_app(APP,DB)
    
    # DB 클래스 정의 모듈 로딩
    from .models import models

    # URL 처리 모듈 연결
    from .views import ageRaingViews
    APP.register_blueprint(ageRaingViews.main_bp)

    # # URL 즉, 클라이언트 요청 페이지 주소를 보여줄 기능 함수
    # def printPage():
    #     return "<h1>HELLO ~</h1>"
    
    # # URL 처리 함수 연결
    @APP.route('/')
    def index():
        results = DB.session.query(models.Anime, models.AgeRating).join(models.AgeRating).order_by(models.Anime.id.desc())
        return render_template('index.html',results=results)

    return APP