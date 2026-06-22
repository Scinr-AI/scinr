"""
tests/unit/test_uid.py — Unit tests for scinr.newton.utils.uid

Imports directly from the submodule to avoid triggering the CLI import chain.
"""
from __future__ import annotations

from scinr.newton.utils.uid import make_instance_uid, make_uid


class TestMakeUid:
    def test_uid_is_string(self):
        """make_uid returns a str."""
        result = make_uid("hello", "world")
        assert isinstance(result, str)

    def test_uid_deterministic(self):
        """Same inputs always produce the same UID."""
        uid1 = make_uid("foo", "bar", "baz")
        uid2 = make_uid("foo", "bar", "baz")
        assert uid1 == uid2

    def test_uid_different_inputs_produce_different_uids(self):
        """Different inputs produce different UIDs."""
        uid_a = make_uid("hello", "world")
        uid_b = make_uid("hello", "earth")
        assert uid_a != uid_b

    def test_uid_format_16_hex_chars(self):
        """make_uid returns exactly 16 lowercase hex characters."""
        uid = make_uid("test")
        assert len(uid) == 16
        assert all(c in "0123456789abcdef" for c in uid)

    def test_uid_single_part(self):
        """make_uid works with a single part."""
        uid = make_uid("only_one")
        assert isinstance(uid, str)
        assert len(uid) == 16

    def test_uid_empty_string_part(self):
        """make_uid handles empty string parts without crashing."""
        uid = make_uid("", "non-empty")
        assert isinstance(uid, str)
        assert len(uid) == 16

    def test_uid_no_collision_with_separator_in_value(self):
        """Length-prefix encoding prevents collisions when values contain separators.

        make_uid("a||b", "c") must differ from make_uid("a", "b||c").
        """
        uid1 = make_uid("a||b", "c")
        uid2 = make_uid("a", "b||c")
        assert uid1 != uid2

    def test_uid_no_collision_colon_separator(self):
        """Collision-free even when values contain ':' characters."""
        uid1 = make_uid("1:2", "3")
        uid2 = make_uid("1", "2:3")
        assert uid1 != uid2

    def test_uid_many_parts(self):
        """make_uid handles many parts correctly."""
        parts = [str(i) for i in range(20)]
        uid = make_uid(*parts)
        assert isinstance(uid, str)
        assert len(uid) == 16

    def test_uid_known_value(self):
        """make_uid produces a stable, known value for a fixed input."""
        import hashlib
        # Manually compute what the UID should be
        parts = ("hello", "world")
        encoded = "||".join(f"{len(p)}:{p}" for p in parts)
        expected = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        assert make_uid("hello", "world") == expected


class TestMakeInstanceUid:
    def test_instance_uid_is_string(self):
        """make_instance_uid returns a str."""
        uid = make_instance_uid("MyModel", {"field": "value"})
        assert isinstance(uid, str)

    def test_instance_uid_deterministic(self):
        """Same model_class + key_fields always produce the same UID."""
        uid1 = make_instance_uid("ConditionModel", {"condition_id": "1", "variation_code": "q.i.a.1(a)"})
        uid2 = make_instance_uid("ConditionModel", {"condition_id": "1", "variation_code": "q.i.a.1(a)"})
        assert uid1 == uid2

    def test_instance_uid_key_order_independent(self):
        """Field insertion order does not affect the UID."""
        uid1 = make_instance_uid("ConditionModel", {"condition_id": "1", "variation_code": "q.i.a.1(a)"})
        uid2 = make_instance_uid("ConditionModel", {"variation_code": "q.i.a.1(a)", "condition_id": "1"})
        assert uid1 == uid2

    def test_instance_uid_different_model_class(self):
        """Different model_class values produce different UIDs."""
        uid1 = make_instance_uid("ModelA", {"key": "value"})
        uid2 = make_instance_uid("ModelB", {"key": "value"})
        assert uid1 != uid2

    def test_instance_uid_different_field_values(self):
        """Different field values produce different UIDs."""
        uid1 = make_instance_uid("MyModel", {"field": "value1"})
        uid2 = make_instance_uid("MyModel", {"field": "value2"})
        assert uid1 != uid2

    def test_instance_uid_format(self):
        """make_instance_uid returns exactly 16 lowercase hex characters."""
        uid = make_instance_uid("SomeModel", {"id": "abc"})
        assert len(uid) == 16
        assert all(c in "0123456789abcdef" for c in uid)
