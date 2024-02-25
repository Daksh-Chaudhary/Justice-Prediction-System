# Justice Prediction System

## Overview

The Justice Prediction System is an Machine Learning application which uses scikit-learn to predict the potential outcome in legal cases based on a dataset sourced from Kaggle. The model takes three inputs: the first party, the second party, and the case description, to forecast the possible winner of the case. The web interface, powered by Streamlit, offers an interactive and user-friendly experience.

### Dataset
The dataset used for training the model can be found on Kaggle. You can access it [here](https://drive.google.com/file/d/1bkOMK_-L0ftzymls4gaNnG7PxqK_mzCD/view?usp=sharing).

### Trained Model
The pre-trained machine learning model can be downloaded from the following link: [Trained Model](https://drive.google.com/file/d/171zXWqQ4KZ-etC6CUw8BUxi6XYqc0-kD/view?usp=sharing).

## Dependencies

Ensure you have the following dependencies installed:

- Python (>=3.6)
- scikit-learn
- streamlit
- pandas
- numpy

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

To run the Justice Prediction System, follow these steps:

1. Open the provided Colab file `Model_training_and_testing.ipynb` in Google Colab.
2. Navigate to the "Web Interface for ML Model" section.
3. Execute the cells in that section to load the pre-trained model and set up the Streamlit web application.
4. Once the setup is complete, run the Streamlit application cells to launch the web interface.

## Usage

1. Upon launching the web application, you will be presented with an intuitive interface.
2. Input the first party, second party, and case description for the legal case to be predicted.
3. Click the "Predict" button to obtain the prediction result.
4. Explore additional functionalities and visualizations available in the application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Justice Prediction System utilizes scikit-learn, Streamlit, pandas, and numpy.
- Special thanks to the open-source community for their contributions to the development of the libraries used in this project.
