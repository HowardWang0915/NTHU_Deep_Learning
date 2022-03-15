# Setup
In order to run the project, make sure you have pytorch and GPU available. Use torch.cuda.is_available() to check if gpu available.

Once you have the hardware available, run
``
pip install -r requirements.txt
``
in order to install the required python packages.

# Training
Now, view the configurations under config.json, make sure that everything is correct. The example given is already runable, you don't have to change anything if you want to just test out the code.
To run training, type
``
python train.py -c config.json
``
Once you finished training, a log and several model(based on how frequent your checkpint is saved) will be saved under the "saved/" folder.
Also, a tensorboard is available. If you want to use the tensorboard, open it under config.json, change "tensorboard: true".
To use the tensorboard, type the following at the project root:
```
tensorboard --logdir saved/log/
```
A tensorboard will be opened at http://localhost:6006

# Generate images
Now, to generate images from the model, run
```
python test.py -c config.json -r checkpoint-epoch300.pth
```
The -r option specifies the checkpoint model. The example "checkpoint-epoch300.pth" is a pretrained model by me, or you can simply replace it by your own model.
As a reminder, once you have finished training, a model will be generated under saved/models/Wafer_Autoencoder/run_id, and you can use that model if you want.

Once you run testing, you will find the .npy file saved inside of log/Wafer_Autoencoder/run_id/. Some visualization
of the gernated images will be stored inside of `img/`

There is already an example gen_data.npy and gen_label.npy under the root folder. To visualize a random sample from each class(note, there are 9 of them), run
```
python visualize.py
```
This will show the generated image on the fly, meaning that the images will not be stored. Note that everytime it show different images, which indicates that
the `.npy` is correctly constructed. Remember to replace the `gen_label.npy` and `gen_data.npy` if you want to visualize your own results.

Lastly, the report is given as hw2report_107061112.pdf, under the root folder.
