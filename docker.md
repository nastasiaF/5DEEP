### Lancer un docker avec Tensorboard pour comparer les logs des models.  

docker pull tensorflow/tensorflow

docker run -it --rm tensorflow/tensorflow bash

docker run -v /Documents/Jupiter-Notebook:/container/project run -it --rm tensorflow/tensorflow bash



docker run -d -p 6006:6006 -v /Documents/Jupiter-Notebook/logs:/container/project run -it --rm tensorflow-tensorboard bash


docker run -d -p 6006:6006 -v $(pwd)/logs:/logs --name my-tf-tensorboard volnet/tensorflow-tensorboard
