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
        - Incorrect prediction? Correct it yourself
        - Save changes
        - Retrain
        - Doubt the predictions? Hit the explain button
        """
    )

    file = st.file_uploader('Upload file', type="csv")
    show_file = st.empty()

    if not file:
        show_file.info("Please upload a valid CSV file")
        return

    df = pd.read_csv(file)
    count = 0

    def predict(df):
        data = df.copy()
        data = data.drop(["url", "label"], axis=1)
        Y = df['label']
        X = data
        if count == 1:
            pickle_in = open("LR_model.pkl", "rb")
        else:
            pickle_in = open("LR_model_new.pkl", "rb")
        clf = pickle.load(pickle_in)
        prediction = clf.predict(X)
        prob = clf.predict_proba(X)
        confidence_score = [prob[i][1] for i in range(len(prob))]  # Confidence in Malicious class
        
        df['prediction'] = prediction
        df['confidence_score'] = confidence_score
        score = clf.score(X, Y)
        st.write('Model Accuracy is:', score)
        return df

    def retrain(df):
        data = df.copy()
        data = data.drop(["url", "confidence_score"], axis=1)
        Y = df['prediction']
        X = data
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
        count += 1
        output_df = predict(df)
        gd = GridOptionsBuilder.from_dataframe(output_df)
        gd.configure_column('prediction', editable=True, type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=0)
        gd.configure_pagination(enabled=True)
        gridoptions = gd.build()
        grid_table = AgGrid(output_df, gridOptions=gridoptions, reload_data=False, data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.MODEL_CHANGED, theme="alpine")
        
        if st.button('Save Changes'):
            new_df = grid_table['data']
            new_df.to_csv('file1.csv', index=False)
            
            st.dataframe(new_df)
            button2 = st.button('Retrain')
            if 'button_state' not in st.session_state:
                st.session_state.button_state = False

            if button2 or st.session_state.button_state:
                st.session_state.button_state = True 
                acc = retrain(new_df)
    
        with st.expander("Don't trust the model predictions? "):
            train_data = df.drop("label",axis = 1)
            train_data = train_data.drop("url", axis =1)
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

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()
