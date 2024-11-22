"""empty message

Revision ID: 3704a2720cf1
Revises: 
Create Date: 2024-10-25 10:00:41.957365

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '3704a2720cf1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('person')
    op.drop_table('answer')
    op.drop_table('question')
    op.drop_table('favorit_food')
    op.drop_table('user')
    op.drop_table('bbs')
    with op.batch_alter_table('age_rating', schema=None) as batch_op:
        batch_op.alter_column('age_id',
               existing_type=mysql.TINYINT(),
               type_=sa.SmallInteger(),
               existing_nullable=False,
               autoincrement=True)
        batch_op.alter_column('age_name',
               existing_type=mysql.VARCHAR(length=20),
               nullable=False)

    with op.batch_alter_table('anime', schema=None) as batch_op:
        batch_op.alter_column('title',
               existing_type=mysql.VARCHAR(length=255),
               nullable=False)
        batch_op.alter_column('story',
               existing_type=mysql.TEXT(),
               nullable=False)
        batch_op.alter_column('age_id',
               existing_type=mysql.TINYINT(),
               type_=sa.Integer(),
               existing_nullable=True)
        batch_op.drop_constraint('anime_ibfk_1', type_='foreignkey')
        batch_op.create_foreign_key(None, 'age_rating', ['age_id'], ['age_id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('anime', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.create_foreign_key('anime_ibfk_1', 'age_rating', ['age_id'], ['age_id'], onupdate='CASCADE', ondelete='CASCADE')
        batch_op.alter_column('age_id',
               existing_type=sa.Integer(),
               type_=mysql.TINYINT(),
               existing_nullable=True)
        batch_op.alter_column('story',
               existing_type=mysql.TEXT(),
               nullable=True)
        batch_op.alter_column('title',
               existing_type=mysql.VARCHAR(length=255),
               nullable=True)

    with op.batch_alter_table('age_rating', schema=None) as batch_op:
        batch_op.alter_column('age_name',
               existing_type=mysql.VARCHAR(length=20),
               nullable=True)
        batch_op.alter_column('age_id',
               existing_type=sa.SmallInteger(),
               type_=mysql.TINYINT(),
               existing_nullable=False,
               autoincrement=True)

    op.create_table('bbs',
    sa.Column('bbsID', mysql.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('bbsTitle', mysql.VARCHAR(length=50), nullable=True),
    sa.Column('userID', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('bbsDate', mysql.DATETIME(), nullable=True),
    sa.Column('bbsContent', mysql.VARCHAR(length=2048), nullable=True),
    sa.Column('bbsAvailable', mysql.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('bbsID'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('user',
    sa.Column('userID', mysql.VARCHAR(length=20), nullable=False),
    sa.Column('userPassword', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('userName', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('userGender', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('userEmail', mysql.VARCHAR(length=50), nullable=True),
    sa.PrimaryKeyConstraint('userID'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('favorit_food',
    sa.Column('person_id', mysql.SMALLINT(unsigned=True), autoincrement=False, nullable=False),
    sa.Column('food', mysql.VARCHAR(length=20), nullable=False),
    sa.ForeignKeyConstraint(['person_id'], ['person.person_id'], name='fk_fav_food_person_id'),
    sa.PrimaryKeyConstraint('person_id', 'food'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('question',
    sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('subject', mysql.VARCHAR(length=200), nullable=False),
    sa.Column('content', mysql.TEXT(), nullable=False),
    sa.Column('create_date', mysql.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('answer',
    sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('question_id', mysql.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('content', mysql.TEXT(), nullable=False),
    sa.Column('create_date', mysql.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['question_id'], ['question.id'], name='answer_ibfk_1', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('person',
    sa.Column('person_id', mysql.SMALLINT(unsigned=True), autoincrement=True, nullable=False),
    sa.Column('fname', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('lname', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('eye_color', mysql.ENUM('BR', 'BL', 'GR'), nullable=True),
    sa.Column('birth_date', sa.DATE(), nullable=True),
    sa.Column('street', mysql.VARCHAR(length=30), nullable=True),
    sa.Column('city', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('state', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('country', mysql.VARCHAR(length=20), nullable=True),
    sa.Column('postal_code', mysql.VARCHAR(length=20), nullable=True),
    sa.PrimaryKeyConstraint('person_id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    # ### end Alembic commands ###