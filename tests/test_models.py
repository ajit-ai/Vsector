import pytest
from vsector.models.vector_record import VectorRecord
from vsector.models.namespace import Namespace

def test_vector_record():
    r = VectorRecord(namespace="ns", vector=[1,2,3], dimension=3, metadata={"a": 1})
    assert r.checksum
    assert not r.is_expired()

def test_namespace_validation():
    ns = Namespace(name="test", dimension=4)
    ns.validate_dimension(4)
    try:
        ns.validate_dimension(5)
        assert False
    except ValueError:
        pass

def test_dimension_mismatch():
    try:
        VectorRecord(namespace="ns", vector=[1,2], dimension=3)
        assert False
    except ValueError:
        pass
