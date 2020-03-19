# RICH-GAN-2019

## Docker 

#### Pulling

```docker pull mrartemev/pytorch```

#### Usage
To run docker image run following code inside repo directory

```docker run --rm -v `pwd`:/workspace --name <name> --gpus all -it -p <port>:8888 <image_name>```

For example:

```docker run --rm -v `pwd`:/workspace --name rich --gpus all -it -p 8888:8888 mrartemev/pytorch```

Running this command will mount docker to your repo directory and execute jupyter notebook command inside your docker.

Open this in your browser to work with repo http://localhost(or yours server-id):8888 (or other chosen <port>).

## Data

###### Real data with s-weights:
`./get_data_calibsample.sh`

###### MC data:
`./get_mc_data_pid.sh`


