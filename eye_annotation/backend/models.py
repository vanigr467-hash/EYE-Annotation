from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from backend.database import Base

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    owner = Column(String, nullable=False)
    label_schema = Column(JSON, default=["pupil", "iris"])
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    media_assets = relationship("MediaAsset", back_populates="project", cascade="all, delete-orphan")

class MediaAsset(Base):
    __tablename__ = "media_assets"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # 'image' or 'video'
    storage_url = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    status = Column(String, default="uploaded", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="media_assets")
    frames = relationship("Frame", back_populates="media_asset", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="media_asset", cascade="all, delete-orphan")

class Frame(Base):
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True, index=True)
    media_asset_id = Column(Integer, ForeignKey("media_assets.id", ondelete="CASCADE"), index=True, nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    storage_url = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    media_asset = relationship("MediaAsset", back_populates="frames")
    annotations = relationship("Annotation", back_populates="frame", cascade="all, delete-orphan")

class Annotation(Base):
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey("frames.id", ondelete="CASCADE"), index=True, nullable=True)
    media_asset_id = Column(Integer, ForeignKey("media_assets.id", ondelete="CASCADE"), index=True, nullable=True)
    label_type = Column(String, nullable=False)
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    rx = Column(Float, nullable=False)
    ry = Column(Float, nullable=False)
    angle = Column(Float, default=0.0)
    boundary_points_json = Column(JSON, nullable=True)
    mask_url = Column(String, nullable=True)
    source = Column(String, nullable=False)  # 'ai' or 'human'
    confidence = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    status = Column(String, default="pending_review", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    frame = relationship("Frame", back_populates="annotations")
    media_asset = relationship("MediaAsset", back_populates="annotations")
    versions = relationship("AnnotationVersion", back_populates="annotation", cascade="all, delete-orphan")

class AnnotationVersion(Base):
    __tablename__ = "annotation_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    annotation_id = Column(Integer, ForeignKey("annotations.id", ondelete="CASCADE"), index=True, nullable=False)
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    rx = Column(Float, nullable=False)
    ry = Column(Float, nullable=False)
    angle = Column(Float, nullable=False)
    boundary_points_json = Column(JSON, nullable=True)
    edited_by = Column(String, nullable=True, default="system")
    action = Column(String, nullable=False)  # 'create', 'edit', 'approve', 'reject'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    annotation = relationship("Annotation", back_populates="versions")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, nullable=True)
    role = Column(String, nullable=False)  # 'admin', 'annotator', 'reviewer', 'viewer'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
