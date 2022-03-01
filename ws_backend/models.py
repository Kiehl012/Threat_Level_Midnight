from database import Base
from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.types import DateTime
from sqlalchemy.dialects.postgresql import ARRAY


class Characters(Base):
    """
    Characters table
    """
    __tablename__ = 'characters'
    emp_id = Column(Integer, primary_key=True)
    emp_name = Column(String(256))
    
    def columns_to_dict(self):
        dict_ = {}
        for key in self.__mapper__.c.keys():
            dict_[key] = getattr(self, key)
        return dict_

class Episodes(Base):
    """
    Episode table
    """
    __tablename__ = 'episodes'
    episode_id = Column(Integer, primary_key=True)
    title = Column(String(256))
    season_no = Column(Integer)
    line_id = Column(Integer)
    episode_no = Column(Integer)
    air_date = Column(Date)
    writer = Column(String)
    director = Column(String)

    def columns_to_dict(self):
        dict_ = {}
        for key in self.__mapper__.c.keys():
            dict_[key] = getattr(self, key)
        return dict_

class Scripts(Base):
    """
    Script table
    """
    __tablename__ = 'scripts'
    line_id = Column(Integer, primary_key=True)
    line = Column(String)
    emp_name = Column(String)
    episode_id = Column(Integer)
    season = Column(Integer)
    episode = Column(Integer)
    sentiment = Column(Float)

    def columns_to_dict(self):
        dict_ = {}
        for key in self.__mapper__.c.keys():
            dict_[key] = getattr(self, key)
        return dict_
