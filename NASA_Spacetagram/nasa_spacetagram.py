import streamlit as st
import requests
from datetime import datetime
import time

# Replace the API key with your own key.
api_key = ''

def fetchAPOD(specific_date='', start_date='', end_date=''):
    URL_APOD = "https://api.nasa.gov/planetary/apod"
    date = specific_date
    start_date = start_date
    end_date = end_date
    params = {
        'api_key':api_key,
        'date':date,
        'start_date':start_date,
        'end_date':end_date,
        'hd':'True'
    }

    #for percent_complete in range(100):
    #    time.sleep(0.03)
    #    my_bar.progress(min(100, percent_complete + 10))
    #my_bar.progress(0)

    response = requests.get(URL_APOD,params=params).json()

    for item in range(len(response)):
        with st.form("my_form"+str(item)):
            st.image(response[item]['url'])
            st.subheader(response[item]['title'] + " - " + response[item]['date'])
            st.write(response[item]['explanation'])
            checkbox_val = st.checkbox("Like")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Clear")
    return

st.header("NASA Spacetagram")
st.write("NASA's Astronomy Picture of the Day. Brought to you by NASA's image' API.")
start_date = st.date_input('Start date')
#end_date = st.date_input('End date')

if st.button('Give me some Hot and Spacey!'):
    fetchAPOD(start_date=start_date)