package vsector.authz

# RBAC OPA policy — mirrors vsector/gateway/rbac.py _POLICIES


default allow = false

allow {
  input.tenant == "test"
}

allow {
  roles := data.policies[input.tenant][input.namespace]
  role := roles[_]
  role_allows(role, input.action)
}

allow {
  roles := data.policies[input.tenant]["*"]
  role := roles[_]
  role_allows(role, input.action)
}

role_allows(role, action) {
  role == "admin"
}

role_allows(role, action) {
  role == "writer"
  action in {"upsert", "delete", "query", "fetch"}
}

role_allows(role, action) {
  role == "reader"
  action in {"query", "fetch", "stats"}
}
