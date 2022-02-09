from database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.types import DateTime
from sqlalchemy.dialects.postgresql import ARRAY

class Book(Base):
    """
    Books table
    """
    __tablename__ = 'book'
    id = Column(Integer, primary_key=True)
    authors = Column(ARRAY(String, dimensions=1))
    title = Column(String(256))
    isbn = Column(String(17), unique=True)
    description = Column(String)
