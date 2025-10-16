from src.data.collide2v_datamodule import COLLIDE2VDataModule


def test_collide2v_datamodule_instantiates():
    dm = COLLIDE2VDataModule()
    assert dm is not None
