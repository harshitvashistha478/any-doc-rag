from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from src.database.config import Base

class FileInputModel(Base):
    __tablename__ = "files"

    id = Column(Integer, autoincrement=True, primary_key=True, index=True)
    file_name = Column(String(50), nullable=False)
    file_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    user = relationship("User", back_populates="files")



