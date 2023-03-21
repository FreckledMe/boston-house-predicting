import streamlit as st
from PIL import Image
from tensorflow import keras

img = Image.open('image/original.jpg')
st.title('Predict price of your dream house in Boston')
st.image(img)

crim = st.number_input(label='Per capita crime rate by town',value=0)
zn = st.number_input(label='Proportion of residential land zoned for lots over 25,000 sq.ft',value=0)
indus = st.number_input(label='proportion of non-retail business acres per town',value=0)
chas = st.radio('Charles River dummy variable (1 if tract bounds river; 0 otherwise)',[0,1])
nox = st.number_input(label='Nitric oxides concentration (parts per 10 million)',value=0)
rm = st.number_input(label='Aerage number of rooms per dwelling',value=0)
age = st.number_input(label='proportion of owner-occupied units built prior to 1940',value=0)
dis = st.number_input(label='Weighted distances to five Boston employment centres',value=0)
rad = st.number_input(label='Index of accessibility to radial highways',value=0)
tax = st.number_input(label='Full-value property-tax rate per $10,000',value=0)
ptratio = st.number_input(label='Pupil-teacher ratio by town',value=0)
b = st.number_input(label='1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',value=0)
lstat = st.number_input(label='% lower status of the population',value=0)

model = keras.models.load_model('./model/boston.h5')
with st.form("key1"):
    # ask for input
    button_check = st.form_submit_button("Predict")
    inputs = [[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]]
if button_check:
    st.text(f'Your dream house price is {round((model.predict(inputs)[0][0])*1000,2)} $')