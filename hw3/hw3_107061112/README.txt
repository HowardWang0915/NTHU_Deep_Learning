First install the environment using
pip install -r requirements.txt

To run training, type
python train.py -c config.json
This will generate a log inside saved/ folder, also will save your checkpoint model.

To run testing, type
python test.py -c config.json -r checkpoint-epoch40.npz
or
python test.py -c config.json -r checkpoint-epoch5.npz

Be sure not to remove config.json and checkpoint-epoch40.npz, otherwise there will be an error.
