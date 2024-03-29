## Test the Model

`predict_real_integrals.py` contains the necessary code to test the an existing model.

### Usage:
```shell
usage: test.py [-h] [--input_image_path INPUT_IMAGE_PATH] [--folder_datatype FOLDER_DATATYPE] [--model_path MODEL_PATH] [--get_info_every GET_INFO_EVERY] [--output_folder OUTPUT_FOLDER] [--make_plots | --no-make_plots] [--generate_outputs | --no-generate_outputs]

Testing interface for SwinSR

options:
  -h, --help                                    show this help message and exit
  --input_image_path INPUT_IMAGE_PATH           can be either a path to a single file or to a directory containing files. default: `test/`
  --folder_datatype FOLDER_DATATYPE             can be either `tiff` or `png`. default: `tiff`
  --model_path MODEL_PATH                       path to the weights of a trained model. default: `out/model.pth`.
  --get_info_every GET_INFO_EVERY               number of iterations that should be done before printing out some results. default: `1000`
  --output_folder OUTPUT_FOLDER                 directory where the output needs to be stored to. default: `out/`
  --make_plots, --no-make_plots                 boolean flag to specify if the plots should be created. deafult: `False`
  --generate_outputs, --no-generate_outputs     boolean flag to specify if the outputs should be created. deafult: `True`
```

## Train the Model

`train.py` contains the necessary code to train the model. Note the model can be trained either in single pass or double pass.
For the difference between these modes please refer to the project report.
The suggested way is to start training the model with the single pass mode and switch to double pass mode only after training for some time. (e.g. after 50000 batches).

### Usage:
```shell
usage: train.py [-h] [--train_path TRAIN_PATH] [--val_path VAL_PATH] [--model_path MODEL_PATH] [--samples_per_update SAMPLES_PER_UPDATE] [--checkpoint_every CHECKPOINT_EVERY] [--multi_pass | --no-multi_pass]

Training interface for SwinSR

options:
  -h, --help            					show this help message and exit
  --train_path TRAIN_PATH					path to the directory that contains training data (.tiff) files. default: train/
  --val_path VAL_PATH						path to the directory that contains validation data (.tiff) files. default: val/
  --model_path MODEL_PATH					path to the weights of a trained model. e.g. out/model.pth.
  --samples_per_update SAMPLES_PER_UPDATE	the number of samples that needs to be used for a single update. default: 16
  --checkpoint_every CHECKPOINT_EVERY		number of samples between every checkpoint. e.g. calculating validation loss, saving the model, etc. default: 200
  --multi_pass, --no-multi_pass				boolean flag to select the training mode. if --multi_pass is specified the model is trained with double path, otherwise single pass.
```

#### Example Usage:
```shell
python train.py --train_path train --val_path val --checkpoint_every 200 --samples_per_update 16 --no-multi_pass
python train.py --train_path train --val_path val --model_path tmp/model.pth --checkpoint_every 200 --samples_per_update 16 --multi_pass
```