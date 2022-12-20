import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
from st_aggrid.shared import GridUpdateMode, DataReturnMode
import pandas as pd

# vpth = “your path” # path that contains the csv file
# csvfl = “tst.csv” # I used a small csv file containing 2 columns: Name & Amt
tdf = pd.read_csv("try_data.csv") # load csv into dataframe

gb = GridOptionsBuilder.from_dataframe(tdf)
gb.configure_column('Name', header_name=('Name'), editable=True)
gb.configure_column('Amt', header_name=('Amount'), editable=True, type=['numericColumn','numberColumnFilter','customNumericFormat'], precision=0)

gridOptions = gb.build()
dta = AgGrid(tdf,
gridOptions=gridOptions,
reload_data=False,
height=200,
editable=True,
theme='streamlit',
data_return_mode=DataReturnMode.AS_INPUT,
update_mode=GridUpdateMode.MODEL_CHANGED)

st.write('Please change an amount to test this')

if st.button('Iterate through aggrid dataset'):
    for i in range(len(dta['data'])): # or you can use for i in range(tdf.shape[0]):
        st.caption(f"df line: {tdf.loc[i][0]} | {tdf.loc[i][1]} || AgGrid line: {dta['data']['Name'][i]} | {dta['data']['Amt'][i]}")

        # check if any change has been done to any cell in any col by writing a caption out
        if tdf.loc[i]['Name'] != dta['data']['Name'][i]:
            st.caption(f"Name column data changed from {tdf.loc[i]['Name']} to {dta['data']['Name'][i]}...")
            # consequently, you can write changes to a database if/as required

        if tdf.loc[i]['Amt'] != dta['data']['Amt'][i]:
            st.caption(f"Amt column data changed from {tdf.loc[i]['Amt']} to {dta['data']['Amt'][i]}...")

tdf = dta['data']    # overwrite df with revised aggrid data; complete dataset at one go
tdf.to_csv('file1.csv', index=False)  # re/write changed data to CSV if/as required
st.dataframe(tdf)    # confirm changes to df