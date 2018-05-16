# QASystem
Question Answering via Machine Comprehension

## Reprocedure:
1. Download dataset:
```bash
sh download.sh
python config.py --mode prepro
```
or download data at [link](https://drive.google.com/open?id=1KUeuSh8U9ZWzHtf3OWIP14XNc6XEcEPb)

2. Add pretrained model
    * Download at: [link](https://drive.google.com/open?id=1n0Dau7nVMaAXU6Sg3n-BIa6dmcGGN4rh)
    * Put all file into > train folder
3. Run example:
```
    python inference.py
```

## Model of RNet ref at:
1. Ref: https://github.com/HKUST-KnowComp/R-Net
