import streamlit as st
import pickle
import numpy as np


rf_model=pickle.load(open('rf_model.pkl','rb'))
#log_model=pickle.load(open('log_model.pkl','rb'))
#svm=pickle.load(open('svc_model.pkl','rb'))

# def classify(num):
#     if num<0.5:
#         return 'Setosa'
#     elif num <1.5:
#         return 'Versicolor'
#     else:
#         return 'Virginica'

def classify(num):
    if num==0:
        return 'CKD'
    else:
        return 'NOT CKD'


def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Chronic Kidney Disease Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # activities=['Linear Regression','Logistic Regression','SVM']
    activities=['Random Forest','KNN','Logistic Regression','Naive Bayes','AdaBoost','XGBoost','Sequential Model','Support Vector Machine']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    # ag=st.slider('Select Age', 10, 80)
    # sw=st.slider('Select Sepal Width', 0.0, 10.0)
    # pl=st.slider('Select Petal Length', 0.0, 10.0)
    # pw=st.slider('Select Petal Width', 0.0, 10.0)
    # inputs=[[sl,sw,pl,pw]]
    
    #age
    age=st.slider('Select Age', 1, 80)
    
    #Blood Pressure
    bp=st.slider('Select Blood Pressure', 0.0, 10.0)

    #specific gravity
    sg=st.slider('Select specific gravity', 0.0, 10.0)

    #albumin
    al=st.slider('Select albumin', 0.0, 10.0)

    #sugar
    su=st.slider('Select sugar', 0.0, 10.0)

    #red blood cells
    rbc=st.slider('Select red blood cells', 0.0, 10.0)

    #pus cell
    pc=st.slider('Select pus cell', 0.0, 10.0)

    #pus cell clumps
    pcc=st.slider('Select pus cell clumps', 0.0, 10.0)

    #bacteria
    ba=st.slider('Select bacteria', 0.0, 10.0)

    #Blood Glucose Random
    bgr=st.slider('Select Blood Glucose Random', 0.0, 10.0)

    #Blood Urea
    bu=st.slider('Select Blood Urea', 0.0, 10.0)

    #Serum Creatinine
    sc=st.slider('Select Serum Creatinine', 0.0, 10.0)

    #Sodium
    sod=st.slider('Select Sodium', 0.0, 10.0)

    #Potassium
    pot=st.slider('Select Potassium', 0.0, 10.0)

    #Hemoglobin
    hemo=st.slider('Select Hemoglobin', 0.0, 10.0)

    #Packed Cell Volume
    pcv=st.slider('Select Packed Cell Volume', 0.0, 10.0)

    #White Blood Cell Count
    wbcc=st.slider('Select White Blood Cell Count', 0.0, 10.0)

    #Red Blood Cell Count
    rbcc=st.slider('Select Red Blood Cell Count', 0.0, 10.0)

    #Hypertension
    htn=st.slider('Select Hypertension', 0.0, 10.0)

    #Diabetes Mellitus
    dm=st.slider('Select Diabetes Mellitus', 0.0, 10.0)

    #Coronary Artery Disease
    cad=st.slider('Select Coronary Artery Disease', 0.0, 10.0)

    #Appetite
    appet=st.slider('Select Appetite', 0.0, 10.0)

    #Pedal Edema
    pe=st.slider('Select Pedal Edema', 0.0, 10.0)

    #Anemia
    ane=st.slider('Select Anemia', 0.0, 10.0)

    
        
    inputs=[[age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn,dm,cad,appet,pe,ane]]
    
    
    
    
    if st.button('Classify'):
        if option=='Random Forest':
            st.success(classify(rf_model.predict(inputs)))
        elif option=='KNN':
            st.success(classify(log_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='Naive Bayes':
            st.success(classify(log_model.predict(inputs)))
        elif option=='AdaBoost':
            st.success(classify(log_model.predict(inputs)))
        elif option=='XGBoost':
            st.success(classify(log_model.predict(inputs)))
        elif option=='Sequential Model':
            st.success(classify(log_model.predict(inputs)))
        else:
           st.success(classify(svm.predict(inputs)))


if __name__=='__main__':
    main()
