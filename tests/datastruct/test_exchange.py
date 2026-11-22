import os
from unittest.mock import MagicMock, patch

import pytest

from datastruct.Exchange import Exchange


class ConcreteExchange(Exchange):
    def populate(self, source):
        self._data = source

def test_exchange_instantiation():
    # Abstract class cannot be instantiated
    with pytest.raises(TypeError):
        Exchange()

def test_concrete_exchange():
    ex = ConcreteExchange()
    assert ex.data is None

    ex.populate("test_data")
    assert ex.data == "test_data"

def test_save_load(tmp_path):
    ex = ConcreteExchange()
    ex.populate({"key": "value"})

    file_path = tmp_path / "test_exchange.pkl"
    ex.save(str(file_path))

    assert os.path.exists(file_path)

    loaded_ex = ConcreteExchange.load(str(file_path))
    assert isinstance(loaded_ex, ConcreteExchange)
    assert loaded_ex.data == {"key": "value"}

def test_fetch_mock():
    # Mock requests.get
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"obj_data": "fetched"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        data = Exchange.fetch({}, "http://example.com/data.json")
        assert data == {"obj_data": "fetched"}
