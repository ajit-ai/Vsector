import pytest
from vsector.gateway.rbac import check_permission


def test_rbac_allow():
    check_permission("test", "any", "upsert")  # admin
    check_permission("demo", "products", "query")  # reader


def test_rbac_deny():
    try:
        check_permission("demo", "products", "delete")  # demo products has writer -> delete not allowed? actually writer allows delete
        # adjust: demo * reader denies upsert on unknown? Let's test strict
        check_permission("unknown", "ns", "upsert")
        assert False, "should deny"
    except Exception as e:
        assert "RBAC deny" in str(e) or "403" in str(e)


def test_opa_policy_exists():
    import pathlib

    assert pathlib.Path("policy/rbac.rego").exists()
    assert "allow" in pathlib.Path("policy/rbac.rego").read_text()
