import pytest

from route_distances.lstm.data import InMemoryTreeDataset, TreeDataModule


@pytest.fixture
def dummy_dataset_input():
    trees = ["tree1", "tree2", "tree3"]
    pairs = [(0, 1, 0.5), (0, 2, 0.7)]
    return pairs, trees


def test_create_dataset(dummy_dataset_input):
    dataset = InMemoryTreeDataset(*dummy_dataset_input)

    assert len(dataset) == 2


def test_dataset_indexing(dummy_dataset_input):
    dataset = InMemoryTreeDataset(*dummy_dataset_input)

    assert dataset[0] == {"tree1": "tree1", "tree2": "tree2", "ted": 0.5}

    assert dataset[1] == {
        "tree1": "tree1",
        "tree2": "tree3",
        "ted": 0.7,
    }

    with pytest.raises(IndexError):
        _ = dataset[2]


def test_setup_datamodule(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, split_part=0.2)

    data.setup()

    assert len(data.train_dataset) == 6
    assert len(data.val_dataset) == 2
    assert len(data.test_dataset) == 2

    assert [idx2 for _, idx2, _ in data.train_dataset.pairs] == [0, 1, 8, 9, 2, 3]
    assert [idx2 for _, idx2, _ in data.val_dataset.pairs] == [6, 7]
    assert [idx2 for _, idx2, _ in data.test_dataset.pairs] == [4, 5]


def test_train_dataloader(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, shuffle=False, split_part=0.2)
    data.setup()

    dataloader = data.train_dataloader()

    assert len(dataloader) == 3

    batches = [batch for batch in dataloader]
    assert len(batches) == 3

    # Do some checks on the structure, but not everything
    assert len(batches[0]["ted"]) == 2
    assert len(batches[0]["tree1"]["tree_sizes"]) == 2
    assert batches[0]["tree1"]["tree_sizes"][0] in [3, 5]
    assert batches[0]["tree1"]["tree_sizes"][1] in [3, 5]


def test_val_and_test_dataloader(shared_datadir):
    pickle_path = str(shared_datadir / "test_data.pickle")
    data = TreeDataModule(pickle_path, batch_size=2, split_part=0.2)
    data.setup()

    assert len(data.val_dataloader()) == 1
    assert len(data.test_dataloader()) == 1
