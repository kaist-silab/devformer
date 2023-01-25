# DPP Benchmark
![example workflow](https://github.com/alstn12088/DPP_benchmark/actions/workflows/pytest.yml/badge.svg) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo is for decap placement problem (DPP) benchmark. 


* How to Run Simulator

```bash
from simulator.decap_sim import decap_sim

cost = decap_sim(probe = 23, solution = [1,5,7], keep_out = [2,3,10])
```

* How to benchmark baselines

```bash
python benchmark.py --baseline [BASELINE NAME]
```

* How to Evalutate pretrained CSE

```bash
python benchmark.py --baseline CSE
```

* How to train CSE

```bash
cd baselines/CSE/

python train.py --num_data 1000 --num_augment 3
```


## DPP Simulator GUI

<p align="center">
    <img src="pages/assets/catchy.png" width="500"/>
</p> 

The application is based on [Streamlit](https://streamlit.io/) which allows for web GUIs in Python. To run the application locally, run the following command:

```bash
streamlit run app.py
```

A web browser should open automatically and you can interact with the application. If it doesn't, you can manually open a browser and navigate to http://localhost:8501.

### Notes on GUI development
The structure of the application is as follows:
```
├── app.py # landing page to `streamlit run`
└── pages/
    ├── about.py # about page in Python (as per Streamlit documentation)
    ├── assets/
    |   └── * # media such as .png images
    └── src/
       └── script.js # javascript file for modifying the GUI
```

Most radical modifications are not supported in Streamlit, so we hack our way and [inject Javascript code](https://www.youtube.com/watch?v=OVgPJEMDkak) to modify elements of the GUI.

### Deploy the app
There are many ways to deploy the app, among which on our own server. However, Streamlit provides a [free hosting service](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app) that is sufficient for our purposes. To deploy the app, simply follow the instructions there or click the "deploy" button after running the app locally!