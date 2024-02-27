[![License](https://img.shields.io/badge/license-MIT-green)](../raac-semantic-segmentation/LICENSE.md)

# RAAC Semantic Segmentation

## Contents
- [Loughborough University RAAC Team](#loughborough-university-raac-team)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Loughborough University RAAC Team
- [Professor Chris Goodier](https://www.lboro.ac.uk/departments/abce/staff/chris-goodier/)
- [Professor Christopher Gorse](https://www.lboro.ac.uk/departments/abce/staff/christopher-gorse/)
- [Professor Sergio Cavalaro](https://www.lboro.ac.uk/departments/abce/staff/sergio-cavalaro/)
- [Dr Karen Blay](https://www.lboro.ac.uk/departments/abce/staff/karen-blay/)
- [Seongha Hwang](https://www.lboro.ac.uk/departments/abce/staff/seongha-hwang/)

## Dataset

You can check the dataset details using the command:

```bash
python check-dataset.py
```
```bash
python check-augmented-dataset.py
```

## Dependencies
We provide a user-friendly configuring method via [Conda](https://docs.conda.io/en/latest/) system, and you can create a new Conda environment using the command:

```bash
conda env create -f environment.yml
```

## Training
Before the training, please download the dataset and copy it into the folder `datasets`.   
Then, you can train the model using the command:

```bash
python train.py --model unet --output output/unet/
# python train.py --model unet++ --output output/unetpp/
# python train.py --model deeplabv3+ --output output/deeplabv3p/
# python train.py --model fpn --output output/fpn/
```

## Inference


## Evaluation


## Acknowledgements


## Citation
If you take use of our datasets or code, please cite our papers:

```
@article{},
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  doi={}
}
```

If you have any questions, please contact me without hesitation (s.hwang3@lboro.ac.uk).

## License
See `LICENSE.md` for more information.


