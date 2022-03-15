First install the environment using
pip install -r requirements.txt

To run training, type
python train.py -c config.json
To run testing, type
python test.py -c config.json -r best.npz

Be sure not to remove config.json and best.npz, otherwise there will be an error.
