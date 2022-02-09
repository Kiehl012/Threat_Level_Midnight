from database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.types import DateTime
from sqlalchemy.dialects.postgresql import ARRAY

class Characters(Base):
    """
    Characters table
    """
    __tablename__ = 'characters'
    emp_id = Column(Integer, primary_key=True)
    emp_name = Column(String(256))

class Episodes(Base):
    """
    Episode table
    """
    __tablename__ = 'episodes'
    episode_id = Column(Integer, primary_key=True)
    title = Column(String(256))
    season_no = Column(Integer)
    episode_no = Column(Integer)

class Scripts(Base):
    """
    Script table
    """
    __tablename__ = 'scripts'
    line_id = Column(Integer, primary_key=True)
    line = Column(String)
    emp_name = Column(String)
    episode_id = Column(Integer)
    sentiment = Column(Float)

