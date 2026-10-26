from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    owner: str = "default"
    label_schema: Optional[List[str]] = ["pupil", "iris"]

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    label_schema: Optional[List[str]] = None

class ProjectResponse(ProjectBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class ProjectList(BaseModel):
    items: List[ProjectResponse]

class MediaAssetBase(BaseModel):
    filename: str
    original_filename: str
    media_type: str
    storage_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_sec: Optional[float] = None
    status: str = "uploaded"

class MediaAssetCreate(MediaAssetBase):
    project_id: int

class MediaAssetResponse(MediaAssetBase):
    id: int
    project_id: int
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class MediaAssetList(BaseModel):
    items: List[MediaAssetResponse]

class FrameBase(BaseModel):
    frame_index: int
    timestamp_sec: float
    storage_url: str
    width: int
    height: int

class FrameResponse(FrameBase):
    id: int
    media_asset_id: int
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class FrameList(BaseModel):
    items: List[FrameResponse]

class AnnotationBase(BaseModel):
    label_type: str
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float = 0.0
    boundary_points_json: Optional[Any] = None
    mask_url: Optional[str] = None
    source: str = "human"
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    created_by: Optional[str] = None
    status: str = "pending_review"

class AnnotationCreate(AnnotationBase):
    frame_id: Optional[int] = None
    media_asset_id: Optional[int] = None

class AnnotationUpdate(BaseModel):
    cx: Optional[float] = None
    cy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    angle: Optional[float] = None
    boundary_points_json: Optional[Any] = None
    status: Optional[str] = None
    edited_by: Optional[str] = "annotator"
    action: Optional[str] = "edit"

class AnnotationResponse(AnnotationBase):
    id: int
    frame_id: Optional[int] = None
    media_asset_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class AnnotationList(BaseModel):
    items: List[AnnotationResponse]

class AnnotationVersionResponse(BaseModel):
    id: int
    annotation_id: int
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float
    boundary_points_json: Optional[Any] = None
    edited_by: Optional[str] = "system"
    action: str
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool = True

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class DetectionRequest(BaseModel):
    image_url: str
    confidence_threshold: Optional[float] = 0.5

class DetectionResponse(BaseModel):
    annotations: List[AnnotationResponse]

class ExportRequest(BaseModel):
    project_id: int
    format: str = "json"

class BatchApproveRequest(BaseModel):
    threshold: float = 85.0
    project_id: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
