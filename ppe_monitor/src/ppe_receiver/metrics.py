"""VSpark-compatible incident routes for the Python receiver."""

import os
from datetime import UTC, datetime
from hmac import compare_digest
from io import BytesIO
from typing import Annotated, Literal
from uuid import UUID, uuid4

import jwt
from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from ppe_receiver.store import Store

ROLE_CLAIM = "http://schemas.microsoft.com/ws/2008/06/identity/claims/role"
MAX_JPEG_BYTES = 100 * 1024 * 1024
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_header = HTTPBearer(auto_error=False)


def has_sa_role(
    key: Annotated[str | None, Security(api_key_header)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_header)],
) -> bool:
    expected = os.getenv("PPE_METRICS_API_KEY")
    if expected and key and compare_digest(expected.encode(), key.encode()):
        return True
    secret = os.getenv("PPE_METRICS_JWT_SECRET")
    if secret and bearer:
        try:
            claims = jwt.decode(
                bearer.credentials, secret, algorithms=["HS256"],
                issuer=os.getenv("PPE_METRICS_JWT_ISSUER"),
                audience=os.getenv("PPE_METRICS_JWT_AUDIENCE"),
                options={"require": ["exp"]},
            )
        except jwt.InvalidTokenError:
            pass
        else:
            for name in ("role", "roles", ROLE_CLAIM):
                roles = claims.get(name)
                if roles == "SA" or (isinstance(roles, list) and "SA" in roles):
                    return True
            return False
    raise HTTPException(401, "Требуется действительный токен или API-ключ")


def require_sa(allowed: Annotated[bool, Depends(has_sa_role)]) -> None:
    if not allowed:
        raise HTTPException(403, "Требуется роль SA")


class IncidentFields(BaseModel):
    camId: Annotated[int, Field(ge=-(2**31), lt=2**31)]
    roomId: Annotated[int, Field(ge=-(2**31), lt=2**31)]
    status: Literal[0, 1, 2]
    type: Literal[0, 1, 2]
    priority: Literal[0, 1, 2]


class CreateIncident(IncidentFields):
    guid: UUID = Field(default_factory=uuid4)
    date: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PatchIncident(IncidentFields):
    guid: UUID


def metrics_router(store: Store) -> APIRouter:
    router = APIRouter(prefix="/api/Metrics", dependencies=[Depends(has_sa_role)])
    write = [Depends(require_sa)]

    @router.get("/incidents", response_model=list[CreateIncident])
    def list_incidents():
        return store.incidents()

    @router.get("/incidents/{guid}", response_model=CreateIncident)
    def get_incident(guid: UUID):
        rows = store.incidents(str(guid))
        if not rows:
            raise HTTPException(404, "Инцидент не найден")
        return rows[0]

    @router.get("/incidents/{guid}/artifact", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
    def get_artifact(guid: UUID):
        data = store.artifact(str(guid))
        if data is None:
            raise HTTPException(404, "Артефакт не найден")
        return Response(data, media_type="image/jpeg")

    @router.post("/send-incident", dependencies=write, response_class=PlainTextResponse)
    def send_incident(incident: Annotated[str, Form()], image: Annotated[UploadFile, File()]):
        try:
            data = CreateIncident.model_validate_json(incident)
        except ValidationError as err:
            raise HTTPException(400, "Некорректные поля incident") from err
        content = image.file.read(MAX_JPEG_BYTES + 1)
        if len(content) > MAX_JPEG_BYTES:
            raise HTTPException(413, "JPEG превышает допустимый размер")
        if content:
            try:
                with Image.open(BytesIO(content), formats=["JPEG"]) as jpeg:
                    jpeg.verify()
            except (OSError, ValueError, Image.DecompressionBombError) as err:
                raise HTTPException(400, "Требуется JPEG") from err
        date = data.date if data.date.tzinfo else data.date.replace(tzinfo=UTC)
        fields = data.model_dump(mode="json")
        fields["date"] = date.astimezone(UTC).isoformat(timespec="microseconds")
        if not store.create_incident(fields, content, generated_date="date" not in data.model_fields_set):
            raise HTTPException(409, "GUID уже существует")
        return str(data.guid)

    @router.patch("/patch-incident", dependencies=write, response_class=PlainTextResponse)
    def patch_incident(incident: PatchIncident):
        if not store.update_incident(incident.model_dump(mode="json")):
            raise HTTPException(400, "Инцидент не найден")
        return "Инцидент обновлён"

    @router.delete("/delete-incident", dependencies=write, response_class=PlainTextResponse)
    def delete_incident(guid: UUID):
        if not store.delete_incident(str(guid)):
            raise HTTPException(400, "Инцидент не найден")
        return "Инцидент удалён"

    return router
