# ChuNom handwritten recognition
Adapted from https://github.com/Holmeyoung/crnn-pytorch

## Installing
```shell
cd /path/to/projects
git clone git@github.com:IHR-Nom/mcrnn-pytorch.git
cd mcrnn-pytorch

pip install -r requirements.txt
```


## Training your own model
### Prepare data
```shell
cd /path/to/datasets/ihr-nomdb
wget https://morphoboid.labri.fr/data/generated_origin.zip
unzip generated_origin.zip

wget https://morphoboid.labri.fr/data/handwritten.zip
unzip handwritten.zip
rm *.zip

cd /path/to/projects/mcrnn-pytorch

# Generating dataset for the Synthetic Nom String dataset
python3 tool/create_dataset.py --out db/gen/train --file /path/to/datasets/ihr-nomdb/generated_origin/train.txt
python3 tool/create_dataset.py --out db/gen/val --file /path/to/datasets/ihr-nomdb/generated_origin/val.txt

# Generating dataset for the Handwriting dataset
python3 tool/create_dataset.py --out db/hw/train --file /path/to/datasets/ihr-nomdb/handwritten/patches_preprocessed/train.txt
python3 tool/create_dataset.py --out db/hw/val --file /path/to/datasets/ihr-nomdb/handwritten/patches_preprocessed/val.txt
```

### Training
```shell
# Firstly, train a MCRNN network on the Synthetic Nom String dataset
# And save the trained model to expr/net.pt file
python3 train.py -train db/generated/train -val db/generated/val --model_out expr/gen.pt --alphabet_out expr/gen.alphabet

# After the model has been trained, we fine-tune 
# from this pretrained model on our Handwriting dataset
python3 train.py -train db/hw/train -val db/hw/val --pretrained_model expr/net.pt --n_epochs 1000 \ 
            --display_interval 50 --val_interval 100 --model_out expr/hw.pt --alphabet_out expr/hw.alphabet
```

### Testing
```shell
python3 demo.py -m expr/hw.pt -a expr/hw.alphabet -i test.jpg
```