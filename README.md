# ConvexNet-BiGRU
ConvexNet-BiGRU

Fingerprint Spoof Detection using ConvexNet and BiGRU
=
Overview
=
This repository contains several implementations of a deep learning model designed to detect spoof fingerprints using ConvexNet and BiGRU (Bidirectional Gated Recurrent Unit) architectures. The model is applied to cross-sensor and intra-sensor scenarios, processing fingerprint data captured with different spoof materials. The patch size varies to evaluate the impact of different data granularity on detection performance.

The following four combinations are implemented:
=
Cross-Sensor:
=
Patch Size: 16x16, 32x32, 48x48, 56x56

Intra-Sensor:
=
Patch Size: 16x16, 32x32, 48x48, 56x56

The dataset used for training and evaluation is collected from LivDet competitions, specifically the LivDet2011, LivDet2013, and LivDet2015 datasets.

Dataset
=
The datasets used in the experiments are publicly available and can be accessed via the following link:

LivDet Registration: https://livdet.org/registration.php


LivDet2011

LivDet2013

LivDet2015

These datasets contain fingerprint images obtained from real and spoof materials, which are used to train the model for detecting fake or spoofed fingerprints.

Project Files:
=

The repository contains the following files:

ConvexNet_BiGRU_loading_with_Cross_sensor_with_patchsize_16x16.ipynb

ConvexNet_BiGRU_loading_with_Cross_sensor_with_patchsize_32x32.ipynb

ConvexNet_BiGRU_loading_with_Cross_sensor_with_patchsize_48x48.ipynb

ConvexNet_BiGRU_loading_with_Cross_sensor_with_patchsize_56x56.ipynb

ConvexNet_BiGRU_loading_with_intra_sensor_with_patchsize_16x16.ipynb

ConvexNet_BiGRU_loading_with_intra_sensor_with_patchsize_32x32.ipynb

ConvexNet_BiGRU_loading_with_intra_sensor_with_patchsize_48x48.ipynb

ConvexNet_BiGRU_loading_with_intra_sensor_with_patchsize_56x56.ipynb

Installation
Prerequisites
=
Ensure that you have the following installed:

Python 3.x

Jupyter Notebook

TensorFlow 2.x

Keras

NumPy

Matplotlib

Scikit-learn

You can install the required packages via pip:

pip install tensorflow numpy matplotlib scikit-learn

Running the Code

Clone this repository to your local machine.

Navigate to the directory containing the Jupyter notebooks.

Open the desired notebook (e.g., ConvexNet_BiGRU_loading_with_Cross_sensor_with_patchsize_16x16.ipynb).

Execute the notebook cells to run the model training and evaluation.

Model Description
=
The core of the model is a hybrid architecture combining ConvexNet and BiGRU:

ConvexNet is used for feature extraction from the fingerprint images.

BiGRU is employed to capture temporal patterns in the sensor data, enhancing the ability to detect spoof fingerprints.

The different patch sizes (16x16, 32x32, 48x48, 56x56) evaluate the effect of input resolution on model performance, with the goal of finding an optimal balance between computational efficiency and accuracy.

Results
=
The results of the experiments are documented in the respective notebooks. They include performance metrics such as accuracy, precision, recall, and F1-score for both cross-sensor and intra-sensor cases. The experiments show how varying patch sizes affect the model's ability to detect spoof fingerprints.
