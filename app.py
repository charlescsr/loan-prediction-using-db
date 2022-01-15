import pandas as pd
import gradio as gr
import pickle
import os
import requests
from sqlalchemy import create_engine

engine = create_engine('sqlite:///loan_data.db', echo=False)
model = pickle.load(open('model.pickle', 'rb'))
df = pd.read_sql('loans', engine)

def predict_status(credit_history, applicantincome, coapplicantincome, loanamount, loan_amount_term):
    data = [credit_history, applicantincome, loanamount, coapplicantincome, loan_amount_term]
    data = pd.Series(data)

    data = data.values.reshape((1, len(df.drop('Loan_Status', axis=1).columns)))
    prediction = model.predict(data)

    # Insert the new row into the database and replace the existing table in the database
    df.loc[len(df)] = [credit_history, applicantincome, loanamount, coapplicantincome, loan_amount_term, prediction[0]]
    df.to_sql('loans', engine, if_exists='replace', index=False)
    df.to_csv('loans.csv', index=False)

    # Send data to FastAPI server
    url = 'http://127.0.0.1:8000/train/'
    with open('loans.csv', 'rb') as f:
        files = {'data': f}
        r = requests.post(url, files=files)

        # Save pickle file from FastAPI server
        if r.status_code == 200:
            with open('model.pickle', 'wb') as f:
                f.write(r.content)
                f.close()
        else:
            print('Error: ', r.status_code)

    # Remove csv
    os.remove('loans.csv')
    
    return 'Granted' if prediction[0] == 1 else 'Denied'

credit_history_list = df['Credit_History'].unique().tolist()
credit_history_list = ['Yes' if int(x) == 1 else 'No' for x in credit_history_list]

gr.Interface(predict_status, inputs=[   gr.inputs.Dropdown(choices=credit_history_list), 
                                        gr.inputs.Slider(minimum=0, maximum=int(df['ApplicantIncome'].max()), step=10),
                                        gr.inputs.Slider(minimum=0, maximum=int(df['CoapplicantIncome'].max()), step=10),
                                        gr.inputs.Slider(minimum=0, maximum=int(df['LoanAmount'].max()), step=10),
                                        gr.inputs.Slider(minimum=0, maximum=int(df['Loan_Amount_Term'].max()), step=10)],
                            outputs='text').launch()