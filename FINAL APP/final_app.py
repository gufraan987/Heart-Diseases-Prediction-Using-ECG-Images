import streamlit as st
#from PIL import Image  # Import Image from Pillow
from Ecg import ECG

# Initialize ECG object
ecg = ECG()

# Get the uploaded image
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    """#### **UPLOADED IMAGE**"""
    # Read the image using Pillow
    #pil_image = Image.open(uploaded_file)

    # Call the getImage method
    #ecg_user_image_read = ecg.getImage(pil_image)
    ecg_user_image_read = ecg.getImage(uploaded_file)
    # Show the image
    st.image(ecg_user_image_read)

    """#### **GRAY SCALE IMAGE**"""
    # Call the convert Grayscale image method
    ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)

    # Create Streamlit Expander for Gray Scale
    my_expander = st.expander(label='Gray SCALE IMAGE')
    with my_expander:
        st.image(ecg_user_gray_image_read)

    """#### **DIVIDING LEADS**"""
    # Call the Divide leads method
    dividing_leads = ecg.DividingLeads(ecg_user_image_read)

    # Streamlit expander for dividing leads
    my_expander1 = st.expander(label='DIVIDING LEAD')
    with my_expander1:
        st.image('Leads_1-12_figure.png')
        st.image('Long_Lead_13_figure.png')

    """#### **PREPROCESSED LEADS**"""
    # Call the preprocessed leads method
    ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)

    # Streamlit expander for preprocessed leads
    my_expander2 = st.expander(label='PREPROCESSED LEAD')
    with my_expander2:
        st.image('Preprossed_Leads_1-12_figure.png')
        st.image('Preprossed_Leads_13_figure.png')

    """#### **EXTRACTING SIGNALS(1-12)**"""
    # Call the signal extraction method
    ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
    my_expander3 = st.expander(label='CONOTUR LEADS')
    with my_expander3:
        st.image('Contour_Leads_1-12_figure.png')

    """#### **CONVERTING TO 1D SIGNAL**"""
    # Call the combine and convert to 1D signal method
    ecg_1dsignal = ecg.CombineConvert1Dsignal()
    my_expander4 = st.expander(label='1D Signals')
    with my_expander4:
        st.write(ecg_1dsignal)

    """#### **PERFORM DIMENSIONALITY REDUCTION**"""
    # Call the dimensionality reduction function
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
    my_expander4 = st.expander(label='Dimensional Reduction')
    with my_expander4:
        st.write(ecg_final)

    """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
    # Call the pretrained ML model for prediction
    ecg_model = ecg.ModelLoad_predict(ecg_final)
    my_expander5 = st.expander(label='PREDICTION')
    with my_expander5:
        st.write(ecg_model)
