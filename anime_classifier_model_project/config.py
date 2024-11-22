import os

# SQLite3 RDBMS 관련
BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'myweb.db'

# DB관련 기능 구현시 사용할 변수
SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:4665@127.0.0.1:3306/testdb'
SQLALCHEMY_TRACK_MODIFICATIONS = False