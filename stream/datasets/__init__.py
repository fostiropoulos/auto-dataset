"""
To generate the imports you can run:
```python
    s = Stream(root_path)
    print('\n'.join([f"from stream.datasets.{_.name} import {_.__name__}" for _ in s.supported_datasets()]))
```

"""
from stream.datasets.aircraft import Aircraft
from stream.datasets.amazon import Amazon
from stream.datasets.apparel import Apparel
from stream.datasets.aptos2019 import Aptos2019
from stream.datasets.art import Art
from stream.datasets.asl import Asl
from stream.datasets.boat import Boat
from stream.datasets.cars import Cars
from stream.datasets.cataract import Cataract
from stream.datasets.celeba import Celeba
from stream.datasets.cifar10 import Cifar10
from stream.datasets.cifar100 import Cifar100
from stream.datasets.cimagenet import CImageNet
from stream.datasets.colorectal import Colorectal
from stream.datasets.concrete import Concrete
from stream.datasets.core50 import Core50
from stream.datasets.cub import Cub
from stream.datasets.deepweedsx import Deepweedsx
from stream.datasets.dermnet import Dermnet
from stream.datasets.dtd import Dtd
from stream.datasets.electronic import Electronic
from stream.datasets.emnist import Emnist
from stream.datasets.eurosat import Eurosat
from stream.datasets.event import Event
from stream.datasets.face import Face
from stream.datasets.fashion import Fashion
from stream.datasets.fer2013 import Fer2013
from stream.datasets.fgvc6 import Fgvc6
from stream.datasets.fish import Fish
from stream.datasets.flowers import Flowers
from stream.datasets.food101 import Food101
from stream.datasets.freiburg import Freiburg
from stream.datasets.galaxy10 import Galaxy10
from stream.datasets.garbage import Garbage
from stream.datasets.gtsrb import Gtsrb
from stream.datasets.ham10000 import Ham10000
from stream.datasets.handwritten import Handwritten
from stream.datasets.histaerial import Histaerial
from stream.datasets.imdb import Imdb
from stream.datasets.inaturalist import Inaturalist
from stream.datasets.indoor import Indoor
from stream.datasets.intel import Intel
from stream.datasets.ip02 import Ip02
from stream.datasets.kermany2018 import Kermany2018
from stream.datasets.kvasircapsule import Kvasircapsule
from stream.datasets.landuse import Landuse
from stream.datasets.lego import Lego
from stream.datasets.malacca import Malacca
from stream.datasets.manga import Manga
from stream.datasets.minerals import Minerals
from stream.datasets.office import Office
from stream.datasets.oriset import Oriset
from stream.datasets.oxford import Oxford
from stream.datasets.pcam import Pcam
from stream.datasets.places365 import Places365
from stream.datasets.planets import Planets
from stream.datasets.plantdoc import Plantdoc
from stream.datasets.pneumonia import Pneumonia
from stream.datasets.pokemon import Pokemon
from stream.datasets.products import Products
from stream.datasets.resisc45 import Resisc45
from stream.datasets.rice import Rice
from stream.datasets.rock import Rock
from stream.datasets.rooms import Rooms
from stream.datasets.rvl import Rvl
from stream.datasets.santa import Santa
from stream.datasets.satellite import Satellite
from stream.datasets.simpsons import Simpsons
from stream.datasets.sketch import Sketch
from stream.datasets.sports import Sports
from stream.datasets.svhn import Svhn
from stream.datasets.textures import Textures
from stream.datasets.tinyimagenet import TinyImagenet
from stream.datasets.vegetable import Vegetable
from stream.datasets.watermarked import Watermarked
from stream.datasets.weather import Weather
from stream.datasets.yelp import Yelp
from stream.datasets.zalando import Zalando
