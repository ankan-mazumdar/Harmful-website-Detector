import numpy as np
import pandas as pd
import streamlit as st
import pickle
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

def main():
    st.title("Automated Data Labelling")
    st.subheader("Malicious URL Detection") 
    st.markdown(
        """
        
        > Let's follow a _six-step_ process:
        - Upload csv file
        - Predict the output
        - Incorrect prediction??? :face_with_rolling_eyes: Correct it yourself :white_check_mark:
        - Save changes
        - Retrain
        - Doubt the predictions? :confused: Hit the explain button


        """
    )
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Predictor API</h2>
    </div>
    """
    file = st.file_uploader('Upload file', type="csv")
    show_file = st.empty()

    if not file:
        show_file.info("Please upload a valid csv file")
        return

    df = pd.read_csv(file)
    # st.dataframe(df)
    # st.info(len(df))
    count = 0
    def predict(df):
        data = df
        data = data.drop("url",axis=1)
        Y = data['label']
        X = data.drop('label', axis=1)
        if count==1:
            pickle_in = open("LR_model.pkl","rb")
        else:
            pickle_in = open("LR_model_new.pkl","rb")
        clf = pickle.load(pickle_in)
        prediction = clf.predict(X)
        prob = clf.predict_proba(X)
        confidence_score = []
        for i in range(len(prob)):
            if prediction[i]==0:
                confidence_score.append(prob[i][0])
            else:
                confidence_score.append(prob[i][1])
        
        df = df.drop('label', axis=1)
        df['prediction'] = prediction
        df['confidence_score'] = confidence_score
        score = clf.score(X, Y)
        st.write('Model Accuracy is : ',score)
        return df

    def retrain(df):
        i = 1
        data = df
        data = data.drop("url",axis=1)
        data = data.drop("confidence_score",axis=1)
        Y = data['prediction']
        X = data.drop('prediction', axis=1)
        clf_new = LogisticRegression(warm_start=True)
        clf_new.fit(X, Y)
        pickle.dump(clf_new, open('LR_model_new.pkl', 'wb'))
        new_score = clf_new.score(X, Y)
        return new_score
        
        

    button = st.button('Predict')
    if 'button_state' not in st.session_state:
        st.session_state.button_state = False

    if button or st.session_state.button_state:
        st.session_state.button_state = True  
        count+=1
        output_df = predict(df)
        gd = GridOptionsBuilder.from_dataframe(output_df)
        gd.configure_column('prediction',editable=True, type=['numericColumn','numberColumnFilter','customNumericFormat'], precision=0)
        gd.configure_pagination(enabled=True)
        # gd.configure_default_column(groupable=True,min_column_width=1)
        gridoptions = gd.build()
        grid_table = AgGrid(output_df, gridOptions=gridoptions, reload_data=False, data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.MODEL_CHANGED, theme="alpine")
        
        # for i in range(len(grid_table['data'])):
          #   if output_df.loc[i]['prediction'] != grid_table['data']['prediction'][i]:
            #    st.caption(Prediction column data )
        # st.write(grid_table)
        #if st.button("Update Changes"):
        #    new_df = grid_table['data']
        #    new_df.to_csv("data/data.csv", index=False)
        #st.dataframe(pd.DataFrame.from_dict(grid_table))
        if st.button('Save Changes'):
            new_df = grid_table['data']    # overwrite df with revised aggrid data; complete dataset at one go
            new_df.to_csv('file1.csv', index=False)  # re/write changed data to CSV if/as required
            
            st.dataframe(new_df)   # confirm changes to df
            button2 = st.button('Retrain')
            if 'button_state' not in st.session_state:
                st.session_state.button_state = False

            if button2 or st.session_state.button_state:
                st.session_state.button_state = True 
                acc = retrain(new_df)
    

        with st.expander("Don't trust the model predictions? "):
            train_data = df.drop("label",1)
            train_data = train_data.drop("url",1)
            pickle_in = open("LR_model.pkl","rb")
            clf = pickle.load(pickle_in)
            explainer = lime_tabular.LimeTabularExplainer(
                training_data = np.array(train_data),
                feature_names = train_data.columns,
                class_names = ['Benign','Malicious'],
                mode = 'classification'
            )
            for i in range(10):
                exp = explainer.explain_instance(
                    data_row = train_data.iloc[i],
                    predict_fn = clf.predict_proba
                )
                fig = exp.as_pyplot_figure()
                st.pyplot(fig=fig,)
        

    file.close()


if __name__=='__main__':
    st.set_page_config(layout="wide")
    main()
