"""Tests for RegisterTables registry system."""

import pytest
from meetasr.register import RegisterTables


class TestRegisterTables:

    def setup_method(self):
        """Fresh registry for each test."""
        self.tables = RegisterTables()

    def test_register_and_retrieve(self):
        @self.tables.register("model_classes", key="test-model")
        class TestModel:
            pass

        assert "test-model" in self.tables.model_classes
        assert self.tables.model_classes["test-model"] is TestModel

    def test_register_uses_class_name_as_default_key(self):
        @self.tables.register("model_classes")
        class MySpecialModel:
            pass

        assert "MySpecialModel" in self.tables.model_classes

    def test_register_invalid_registry_raises(self):
        with pytest.raises(AttributeError):
            @self.tables.register("nonexistent_registry", key="x")
            class Foo:
                pass

    def test_list_registered_sorted(self):
        @self.tables.register("model_classes", key="zzz")
        class Z:
            pass

        @self.tables.register("model_classes", key="aaa")
        class A:
            pass

        keys = self.tables.list_registered("model_classes")
        assert keys[0] == "aaa"
        assert keys[-1] == "zzz"

    def test_decorator_returns_original_class(self):
        @self.tables.register("model_classes", key="mymodel")
        class MyModel:
            x = 42

        assert MyModel.x == 42   # class is unmodified

    def test_reregister_overwrites(self):
        @self.tables.register("model_classes", key="conflict")
        class First:
            pass

        @self.tables.register("model_classes", key="conflict")
        class Second:
            pass

        assert self.tables.model_classes["conflict"] is Second
