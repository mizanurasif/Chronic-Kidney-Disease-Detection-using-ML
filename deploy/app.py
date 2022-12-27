import streamlit as st
import pickle
import numpy as np


# rf_model=pickle.load(open('rf_model.pkl','rb'))

svm=pickle.load(open('svm.pkl','rb'))
knn=pickle.load(open('knn.pkl','rb'))
lr=pickle.load(open('lr.pkl','rb'))
xgb=pickle.load(open('xgb.pkl','rb'))
rf=pickle.load(open('rf.pkl','rb'))
adab=pickle.load(open('abab.pkl','rb'))
eclf=pickle.load(open('eclf.pkl','rb'))


def classify(prediction):
    print("predicted value: ",prediction)
    if (prediction[0] == 0):
        return 'CKD'
    else:
        return 'NOT CKD'


def main():
    # st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:maroon ;padding:10px">
        <h2 style="color:white;text-align:center;">Chronic Kidney Disease Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # activities=['Linear Regression','Logistic Regression','SVM']
    activities=['Support Vector Machine','KNN','Logistic Regression','XGBoost','Random Forest','AdaBoost','Majority Voting']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    

    # 1 age
    age = st.number_input('Select age',min_value=2,max_value=90,value=30,step=1)
    # age = float(age)
    # st.write('The current age is ', age)


    # 2 bp
    bp = st.number_input('Select bp',min_value=50,max_value=180,value=110,step=1)
    # bp = float(bp)
    # st.write('The current bp is ', bp)


    # 3 sg
    page_sg_options = ['1.005','1.010','1.015','1.020','1.025']
    page_sg = st.radio('**Specific Gravity sg**',page_sg_options, horizontal=1)
    # st.write("**The variable 'page_sg' returns:**",  page_sg)
    # here sg values are 1.005,1.010,1.015,1.020,1.025
    sg = float(page_sg)
    # st.write("**The variable 'sg' returns:**",  sg)


    # 4 al
    al=st.slider('Select Albumin', 0, 5)
    # st.write("**The variable 'al' returns:**",  al)


    # 5 su
    su=st.slider('Select Sugar', 0, 5)
    # st.write("**The variable 'su' returns:**",  su)


    # 6 rbc
    page_rbc_options = ['normal','abnormal']
    page_rbc = st.radio('**Red Blodd Cell RBC**',page_rbc_options, horizontal=1)
    # st.write("**The variable 'page_rbc' returns:**",  page_rbc)
    # here rbc value normal=0 and abnormal=1
    if page_rbc == 'normal':
        rbc = 0
    else:
        rbc = 1


    # 7 pc
    page_pc_options = ['normal','abnormal']
    page_pc = st.radio('**Pus Cell PC**',page_pc_options, horizontal=1)
    # st.write("**The variable 'page_pc' returns:**",  page_pc)
    # here pc value normal=0 and abnormal=1
    if page_pc == 'normal':
        pc = 0
    else:
        pc = 1


    # 8 pcc
    page_pcc_options = ['present','notpresent']
    page_pcc = st.radio('**Pus Cell Clumps PCC**',page_pcc_options, horizontal=1)
    # st.write("**The variable 'page_pcc' returns:**",  page_pcc)
    # here pcc value present=0 and notpresent=1
    if page_pcc == 'present':
        pcc = 0
    else:
        pcc = 1


    # 9 ba
    page_ba_options = ['present','notpresent']
    page_ba = st.radio('**Bacteria BA**',page_ba_options, horizontal=1)
    # st.write("**The variable 'page_ba' returns:**",  page_ba)
    # here ba value present=0 and notpresent=1
    if page_ba == 'present':
        ba = 0
    else:
        ba = 1

    # 10 bgr
    bgr = st.number_input('Select bgr',min_value=22,max_value=490,value=110,step=1)
    # bgr = float(bgr)
    # st.write('The current bgr is ', bgr)


    # 11 bu
    bu = st.number_input('Select bu',min_value=1,max_value=391,value=200,step=1)
    # bu = float(bu)
    # st.write('The current bu is ', bu)    


    # 12 sc
    sc = st.number_input('Select sc',min_value=0,max_value=76,value=40,step=1)
    # sc = float(sc)
    # st.write('The current sc is ', sc)   


    # 13 sod
    sod = st.number_input('Select sod',min_value=4.5,max_value=163.0,value=90.0,step=0.5)
    # sod = float(sod)
    # st.write('The current sod is ', sod)           


    # 14 pot
    pot = st.number_input('Select pot',min_value=2.5,max_value=47.0,value=25.0,step=0.5)
    # pot = float(pot)
    # st.write('The current pot is ', pot)   
        

    # 15 hemo
    hemo = st.number_input('Select hemo',min_value=3.1,max_value=17.8,value=13.0,step=0.1)
    # hemo = float(hemo)
    # st.write('The current hemo is ', hemo)   


    # 16 pcv
    pcv = st.number_input('Select pcv',min_value=0,max_value=54,value=30,step=1)
    # pcv = float(pcv)
    # st.write('The current pcv is ', pcv)   
        

    # 17 wbcc
    wbcc = st.number_input('Select wbcc',min_value=0,max_value=26400,value=10400,step=100)
    # wbcc = float(wbcc)
    # wbcc = wbcc*100
    # st.write('The current wbcc is ', wbcc)   
        

    # 18 rbcc
    rbcc = st.number_input('Select rbcc',min_value=0.0,max_value=8.8,value=2.0,step=0.1)
    # rbcc = float(rbcc)
    # st.write('The current rbcc is ', rbcc)   


    # 19 htn
    page_htn_options = ['yes','no']
    page_htn = st.radio('**Hypertension htn**',page_htn_options, horizontal=1)
    # st.write("**The variable 'page_htn' returns:**",  page_htn)
    # here htn value yes=0 and no=1
    if page_htn == 'yes':
        htn = 0
    else:
        htn = 1


    # 20 dm
    page_dm_options = ['yes','no']
    page_dm = st.radio('**Diabetes Mellitus dm**',page_dm_options, horizontal=1)
    # st.write("**The variable 'page_dm' returns:**",  page_dm)
    # here dm value normal=0 and abnormal=1
    if page_dm == 'yes':
        dm = 0
    else:
        dm = 1



    # 21 cad
    page_cad_options = ['yes','no']
    page_cad = st.radio('**Coronary Artery Disease cad**',page_cad_options, horizontal=1)
    # st.write("**The variable 'page_cad' returns:**",  page_cad)
    # here cad value yes=0 and no=1
    if page_cad == 'yes':
        cad = 0
    else:
        cad = 1



    # 22 appet
    page_appet_options = ['good','poor']
    page_appet = st.radio('**Appetite appet**',page_appet_options, horizontal=1)
    # st.write("**The variable 'page_appet' returns:**",  page_appet)
    # here appet value poor=0 and good=1
    if page_appet == 'poor':
        appet = 0
    else:
        appet = 1



    # 23 pe
    page_pe_options = ['yes','no']
    page_pe = st.radio('**Pedal Edema pe**',page_pe_options, horizontal=1)
    # st.write("**The variable 'page_pe' returns:**",  page_pe)
    # here pe value yes=0 and no=1
    if page_pe == 'yes':
        pe = 0
    else:
        pe = 1


    # 24 ane
    page_ane_options = ['yes','no']
    page_ane = st.radio('**Anemia ane**',page_ane_options, horizontal=1)
    # st.write("**The variable 'page_ane' returns:**",  page_ane)
    # here ane value yes=0 and no=1
    if page_ane == 'yes':
        ane = 0
    else:
        ane = 1


    
    input_data=[age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn,dm,cad,appet,pe,ane]
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    inputs = input_data_as_numpy_array.reshape(1,-1)

    # inputs=np.array([[age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn,dm,cad,appet,pe,ane]])
    
    print(inputs)
    print("type of array inputs: ",inputs.dtype)
    st.write(inputs)

    a = np.array([[48,80,1.02,1,0,0,0,1,1,121,36,1,135.0,3.5,15.4,44,7800,5.2,0,0,1,1,1,1]])
    print(a)
    print("type of array a: ",a.dtype)

    b = np.array([[57,60,1.02,0,0,0,0,1,1,105,49,1,150.0,4.7,15.7,44,10400,6.2,1,1,1,1,1,1]])
    print(b)
    print("type of array inputs: ",b.dtype)




    # print("typ of age: ",type(age))
    # print("typ of bp: ",type(bp))
    # print("typ of sg: ",type(sg))
    # print("typ of al: ",type(al))
    # print("typ of su: ",type(su))
    # print("typ of rbc: ",type(rbc))
    # print("typ of pc: ",type(pc))
    # print("typ of pcc: ",type(pcc))
    # print("typ of ba: ",type(ba))
    # print("typ of bgr: ",type(bgr))
    # print("typ of bu: ",type(bu))
    # print("typ of sc: ",type(sc))
    # print("typ of sod: ",type(sod))
    # print("typ of pot: ",type(pot))
    # print("typ of hemo: ",type(hemo))
    # print("typ of pcv: ",type(pcv))
    # print("typ of wbcc: ",type(wbcc))
    # print("typ of rbcc: ",type(rbcc))
    # print("typ of htn: ",type(htn))
    # print("typ of dm: ",type(dm))
    # print("typ of cad: ",type(cad))
    # print("typ of appet: ",type(appet))
    # print("typ of pe: ",type(pe))
    # print("typ of ane: ",type(ane))

    
    if st.button('Classify'):
        if option=='Random Forest':
            st.success(classify(rf.predict(inputs)))
        elif option=='KNN':
            st.success(classify(knn.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(lr.predict(inputs)))
        elif option=='XGBoost':
            st.success(classify(xgb.predict(inputs)))
        elif option=='AdaBoost':
            st.success(classify(adab.predict(inputs)))
        elif option=='Support Vector Machine':
            st.success(classify(svm.predict(inputs)))
        elif option=='Majority Voting':
            st.success(classify(eclf.predict(b)))

        # elif option=='Naive Bayes':
        #     st.success(classify(nv.predict(inputs)))
            
        # else:
        #    st.success(classify(sm.predict(inputs)))


if __name__=='__main__':
    main()
