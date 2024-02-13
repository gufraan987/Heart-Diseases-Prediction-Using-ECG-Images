# Heart Disease Prediction using ECG Images

#### Abstract:
This project aims to predict heart diseases using electrocardiogram (ECG) images through machine learning models. ECG signals are widely used for diagnosing various heart conditions. By leveraging machine learning techniques, we can automate the process of detecting abnormalities in ECG signals, which can assist healthcare professionals in making accurate diagnoses.

#### Dataset
ECG images: https://data.mendeley.com/datasets/gwbz3fsgp8/2

The dataset used in this project consists of ECG images collected from above given link. It includes ECG recordings from individuals with and without heart diseases. Each ECG image is associated with a binary label indicating the presence or absence of a heart condition.

#### Approach:
The user uploads an ECG image to our web app. Then, we use techniques like rgb2gray conversion, gaussian filtering, resizing, and thresholding to extract only the signals that do not have grid lines. The required waves (P, QRS, T) are then extracted using contour techniques and converted to a 1D signal. The normalized 1D signal is then fed into our pre-trained ML model, which is then analyzed. When the model has completed the analysis, it returns the results to the user based on the findings.

Here, we have used 4 categories for image classification for our ECG images. Normal, Myocardial infarction, Abnormal Heart beat and History of Myocardial infarction.

One benefit of our app is that the user can view the entire workflow in the UI and receive real-time feedback.

The tricky path here is feature extraction from images; if done correctly and optimally, the accuracy of our model can be increased.

#### DEPLOYMENT LINK:
We have deployed our applicaiton in Streamlit.io.

Link: https://heart-diseases-prediction-using-ecg-images-htprsuyxmzmb5iktkds.streamlit.app/

Download any image from the below folder path and try uploading it to the above url to get real-time insights.
https://github.com/gufraan987/Heart-Diseases-Prediction-Using-ECG-Images/tree/main/ECG_IMAGES_DATASET
