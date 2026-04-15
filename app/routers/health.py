"""Health router — liveness/readiness probe."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe — returns 200 when the server is running."""
    return {"status": "ok"}
