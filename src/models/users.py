from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from src.database.config import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    task_id = Column(Integer, nullable=True)
    
    files = relationship("FileInputModel", back_populates="user")