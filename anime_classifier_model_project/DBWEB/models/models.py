# ----------------------------------------------------------------------------------
# 데이터베이스의 테이블 정의 클래스
# ----------------------------------------------------------------------------------
# 모듈로딩
from DBWEB import DB

# ----------------------------------------------------------------------------------
# Question 테이블 정의 클래스
# ----------------------------------------------------------------------------------
class AgeRating(DB.Model):
    __tablename__ = 'age_rating'
    age_id = DB.Column(DB.SmallInteger, primary_key=True, autoincrement=True)
    age_name = DB.Column(DB.String(20), nullable=False)

class Anime(DB.Model):
    # 컬럼 정의
    __tablename__ = 'anime'
    id = DB.Column(DB.Integer, primary_key=True, autoincrement=True)
    title = DB.Column(DB.String(255), nullable=False)
    story = DB.Column(DB.Text(), nullable=False)
    age_id = DB.Column(DB.Integer, DB.ForeignKey('age_rating.age_id'))  # Foreign Key 설정
    # Relationship 설정
    age = DB.relationship('AgeRating', backref=DB.backref('answer_set'))