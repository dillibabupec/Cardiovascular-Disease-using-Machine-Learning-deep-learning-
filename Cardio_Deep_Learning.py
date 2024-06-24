import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
from Ecg import  ECG

st.set_page_config(
    page_title="Cardio Disease Detection",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
components.html(
    """
    <style>
        #effect{
            margin:0px;
            padding:0px;
            font-family: "Source Sans Pro", sans-serif;
            font-size: max(8vw, 20px);
            font-weight: 700;
            top: 0px;
            right: 25%;
            position: fixed;
            background: -webkit-linear-gradient(0.25turn,#FF4C4B, #FFFB80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p{
            font-size: 2rem;
        }
    </style>
    <p id="effect">Cardio Disease Detection Using Deep Learning</p>
    """,
    height=250,
)


def page_layout():
    st.write(" ")

    
#intialize ecg object
ecg = ECG()
#get the uploaded image
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  """#### **UPLOADED IMAGE**"""
  # call the getimage method
  ecg_user_image_read = ecg.getImage(uploaded_file)
  #show the image
  st.image(ecg_user_image_read)

  """#### **GRAY SCALE IMAGE**"""
  #call the convert Grayscale image method
  ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
  
  #create Streamlit Expander for Gray Scale
  my_expander = st.expander(label='Gray SCALE IMAGE')
  with my_expander: 
    st.image(ecg_user_gray_image_read)
  
  """#### **DIVIDING LEADS**"""
   #call the Divide leads method
  dividing_leads=ecg.DividingLeads(ecg_user_image_read)

  #streamlit expander for dividing leads
  my_expander1 = st.expander(label='DIVIDING LEAD')
  with my_expander1:
    st.image('Leads_1-12_figure.png')
    st.image('Long_Lead_13_figure.png')
  
  """#### **PREPROCESSED LEADS**"""
  #call the preprocessed leads method
  ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)

  #streamlit expander for preprocessed leads
  my_expander2 = st.expander(label='PREPROCESSED LEAD')
  with my_expander2:
    st.image('Preprossed_Leads_1-12_figure.png')
    st.image('Preprossed_Leads_13_figure.png')
  
  """#### **EXTRACTING SIGNALS(1-12)**"""
  #call the sognal extraction method
  ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
  my_expander3 = st.expander(label='CONOTUR LEADS')
  with my_expander3:
    st.image('Contour_Leads_1-12_figure.png')
  
  """#### **CONVERTING TO 1D SIGNAL**"""
  #call the combine and conver to 1D signal method
  ecg_1dsignal = ecg.CombineConvert1Dsignal()
  my_expander4 = st.expander(label='1D Signals')
  with my_expander4:
    st.write(ecg_1dsignal)
    
  """#### **PERFORM DIMENSINALITY REDUCTION**"""
  #call the dimensinality reduction funciton
  ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
  my_expander4 = st.expander(label='Dimensional Reduction')
  with my_expander4:
    st.write(ecg_final)
  
  """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
  #call the Pretrainsed ML model for prediction
  ecg_model=ecg.ModelLoad_predict(ecg_final)
  my_expander5 = st.expander(label='PREDICTION')
  with my_expander5:
    st.write(ecg_model)



# Render page layout
page_layout()


