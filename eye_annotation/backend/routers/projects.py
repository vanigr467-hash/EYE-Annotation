from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Any
from backend.database import get_db
from backend.models import Project
from backend.schemas import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("/", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    existing = db.query(Project).filter(Project.name == project.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project '{project.name}' already exists"
        )
    data = project.model_dump() if hasattr(project, "model_dump") else project.dict()
    db_project = Project(**data)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/{id}", response_model=ProjectResponse)
def get_project(id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.put("/{id}", response_model=ProjectResponse)
def update_project(id: int, project_update: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    data = project_update.model_dump(exclude_unset=True) if hasattr(project_update, "model_dump") else project_update.dict(exclude_unset=True)
    for key, value in data.items():
        setattr(db_project, key, value)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(db_project)
    db.commit()
