import streamlit as st
import pickle 
import numpy as np

def conv(arg):
    if arg == 'Yes':
        return 1
    else:
        return 0

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Laptop Predictor")

#Brand
brand = st.selectbox('Brand',df['Company'].unique())

#Type 
type = st.selectbox('Type',df['TypeName'].unique())

#Ram
ram = st.selectbox('Ram(in GB)',[2,6,8,12,16,24,32,64])

#Weight
weight = st.number_input('Weight of the Laptop')

#Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

#ips
ips = st.selectbox('IPS',['No','Yes'])

#Screen size 
screen_size = st.number_input('Screen Size')

#Resolution
available_r = ['1920x1080','1366x768','1600x900','3840x1800','2880x1800','2560x1600','2560x1440','2304x1440']
resolution = st.selectbox('Resolution',available_r)

#Cpu 
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())


#HDD
hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

#ssd
ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

#GPU
gpu_brand = st.selectbox('GPU',df['gpu_brand'].unique())



os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # query
    touchscreen=conv(touchscreen)
    ips = conv(ips)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size
    query = np.array([brand,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu_brand,os])

    query = query.reshape(1,12)
    price = round(int(np.exp(model.predict(query)[0])))

    st.title(f"Predicted Price: {price}")