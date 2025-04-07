from fastapi import APIRouter, UploadFile

router = APIRouter(tags=["Parcellatio"])


@router.post("/parcellation")
def aprcellation(nii_file: UploadFile):
    breakpoint()
