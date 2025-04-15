# Project1

This repository contains the source code for our Neurips submission with the title 'Fine-Tuning Personalization in Federated Learning to Mitigate Adversarial Clients' .

## Installation


1. Navigate to the project directory:

    ```shell
    cd PBML
    ```

2. Install the required modules:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the `src` folder:

    ```shell
    cd src
    ```

2. Run the main script with the desired command line arguments:

```shell
python3 main.py \
--dataset mnist \
--heterogeneity dirichlet_mnist \
--n 20 \
--m 64 \
--f 6 \
--T 5 \
--model cnn \
--lr 0.05 \
--attack SF \
--batch_size 64 \
--nb_main_client 2 \
--nb_run 5 \
--alpha 3.0 \
--nb_classes 10
```
    
3. The results will be saved under experiments folder. To combine multiple results in one figure, use replot_batch.py script (edit the part after if if __name__ == "__main__"). 


## Reproducibility 

To reproduce the mnsit results, run the following command:

```shell
run_mnist.py
``` 

To reproduce the phishing results, run the following command:

```shell
run_phishing.py
```

The seeds are fixed in the scripts allow the exact reproduction of the results.
