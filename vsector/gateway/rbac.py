"""RBAC/OPA per-tenant + mTLS — extends gateway/auth.py.

Roles: admin, writer, reader per namespace. OPA stub via policy JSON.
mTLS: X-Client-Cert header → tenant extraction (in prod: client cert verify).
"""

from __future__ import annotations

import logging
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# In-memory policy store: tenant -> namespace -> roles
# In prod: OPA sidecar at http://opa:8181/v1/data/vsector/allow
_POLICIES: dict[str, dict[str, list[str]]] = {
    "demo": {"products": ["reader", "writer"], "*": ["reader"]},
    "test": {"*": ["admin"]},
}

# mTLS stub: map cert fingerprint → tenant
_MTLS_CERTS: dict[str, str] = {
    "demo-cert-fingerprint": "demo",
}


def _role_allows(role: str, action: str) -> bool:
    table = {"reader": {"query", "fetch", "stats"}, "writer": {"upsert", "delete", "query", "fetch"}, "admin": {"*"} }
    allowed = table.get(role, set())
    return "*" in allowed or action in allowed


def check_permission(tenant: str, namespace: str, action: str) -> None:
    """Raise 403 if not allowed. OPA style: allow if any role permits action.

    In prod, replace with httpx POST to OPA.
    """
    # try OPA if endpoint set
    import os

    opa_url = os.getenv("VSECTOR_OPA_URL")
    if opa_url:
        try:
            import httpx

            r = httpx.post(f"{opa_url}/v1/data/vsector/allow", json={"input": {"tenant": tenant, "namespace": namespace, "action": action}}, timeout=0.2)
            if r.json().get("result", {}).get("allow"):
                return
            raise HTTPException(status_code=403, detail=f"OPA deny {tenant}/{namespace}:{action}")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"OPA unavailable, fallback to local: {e}")

    # local fallback
    ns_policies = _POLICIES.get(tenant, {})
    roles = ns_policies.get(namespace) or ns_policies.get("*") or []
    for role in roles:
        if _role_allows(role, action):
            return
    raise HTTPException(status_code=403, detail=f"RBAC deny tenant={tenant} ns={namespace} action={action}")


def extract_tenant_mtls(x_client_cert: str | None = Header(default=None, alias="X-Client-Cert")) -> str | None:
    """mTLS: extract tenant from client cert fingerprint header."""
    if not x_client_cert:
        return None
    # In prod, verify cert chain + extract CN/SAN → tenant
    tenant = _MTLS_CERTS.get(x_client_cert)
    if tenant:
        return tenant
    # fallback: cert header itself is tenant for dev
    return x_client_cert[:16]


def require_role(action: str):
    """FastAPI dependency factory for RBAC."""

    def _dep(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
        x_client_cert: str | None = Header(default=None, alias="X-Client-Cert"),
    ):
        from .auth import verify_api_key

        # reuse existing auth
        auth = verify_api_key(x_api_key, authorization)  # type: ignore
        tenant = auth.get("sub") or auth.get("tenant") or "dev"
        # mTLS overrides tenant if present
        mtls_tenant = extract_tenant_mtls(x_client_cert)
        if mtls_tenant:
            tenant = mtls_tenant
        # namespace not known here — check will be per-endpoint with namespace param
        # For generic, allow; per-endpoint will call check_permission with namespace
        return {"tenant": tenant, "action": action}

    return _dep
