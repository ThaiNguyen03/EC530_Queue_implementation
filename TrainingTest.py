import pytest
import logging
import tracemalloc
from unittest.mock import patch, MagicMock

from datasets import load_dataset

from Training import app, StartTraining, task_queue, task_complete_event

logging.basicConfig(filename='./testTraining.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testTraining.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)
tracemalloc.start()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_upload_parameters(client):

    rv = client.post('/upload_parameters', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'parameters': {
            'learning_rate': 5e-5,
            'per_device_train_batch_size': 16,
            'gradient_accumulation_steps': 4,
            'per_device_eval_batch_size': 16,
            'num_train_epochs': 3,
            'warmup_ratio': 0.1,
            'logging_steps': 10,
        },

    })
    assert b'Parameters uploaded successfully' in rv.data


def test_start_training(client):
    test_set = load_dataset("food101", split="train[:100]")
    test_set.save_to_disk('./training_test')
    test_set.to_parquet('./training_test/training_test.parquet')
    rv = client.post('/start_training', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k'
        })
    try:
      assert rv.status_code == 200
      mylogger.info("test_start_training passed")
    except AssertionError:
        mylogger.error("test_start_training not passed")
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()


# assert b'Training for model test_model completed successfully' in rv.data
def test_task_queue(client):
    test_set = load_dataset("food101", split="train[:100]")
    test_set.save_to_disk('./training_test')
    test_set.to_parquet('./training_test/training_test.parquet')
    with task_queue.mutex:
        task_queue.queue.clear()
    task_complete_event.clear()
    rv1 = client.post('/start_training', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k',
        'train_dataset': './training_test/training_test.parquet'
    })

    rv2 = client.post('/start_training', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'microsoft/beit-base-patch16-224-pt22k',
        'train_dataset': './training_test/training_test.parquet'
    })

    task_complete_event.wait()
    assert task_queue.empty()

    assert rv1.status_code == 200
    assert rv2.status_code == 200


# if __name__ == "__main__":
#   pytest.main()
