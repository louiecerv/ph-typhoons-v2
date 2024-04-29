import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import time
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam

if "data_norm" not in st.session_state:
    st.session_state.data_norm = None

if "model" not in st.session_state:
    st.session_state.model = None

def app():
    st.subheader('RNN-LSTM Based Typhoon Prediction in the Philippines')
    
    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering)
    \nCCS 229 - Intelligent Systems
    *Department of Computer Science
    *College of Information and Communications Technology
    *##West Visayas State University##"""
    st.text(text)

    text = """This Streamlit app utilizes a bi-directional Recurrent Neural Network 
    (RNN) with Long Short-Term Memory (LSTM) units to analyze historical typhoon 
    data and forecast the likelihood of typhoons affecting the Philippines in a 
    given month. Users can interact with the app to visualize past typhoon 
    patterns and receive monthly forecasts, potentially aiding in disaster 
    preparedness efforts."""
    st.write(text)

    text = """The data is obtained from the following site : 
    https://en.wikipedia.org/wiki/List_of_typhoons_in_the_Philippines_(2000%E2%80%93present)"""
    st.write(text)  

    df = pd.read_csv('./PH-TYPHOONS2000-2023.csv', header=0)
    df['date'] = pd.to_datetime(df['Month'])
    df.drop(columns=['Month'], inplace=True)    

    # Set the 'date' column as the index
    df.set_index('date', inplace=True)

    with st.expander("Show Dataset"):
        st.write("The TIme Series Dataset")
        st.write(df)   
        st.write(df.shape)

    st.write("The Typhoon Count 2000-2023")

    fig, ax = plt.subplots()  
    ax.plot(df['Typhoons'])
    ax.set_title('Typhoons Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('No. of Typhoons')
    ax.grid(True)  
    ax.tick_params(axis='x', rotation=45)   
    st.pyplot(fig)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = StandardScaler()
    data_norm = scaler.fit_transform(df.iloc[:,0].values.reshape(-1, 1))
    data_norm = pd.DataFrame(data_norm)
    st.session_state.data_norm = data_norm

    # Split the data into training and testing sets
    train_size = int(len(data_norm) * 0.8)
    test_size = len(data_norm) - train_size
    train_data, test_data = data_norm.iloc[0:train_size], data_norm.iloc[train_size:len(data_norm)]

    # Convert the data to numpy arrays
    x_train, y_train = train_data.iloc[:-1], train_data.iloc[1:]
    x_test, y_test = test_data.iloc[:-1], test_data.iloc[1:]

    # Reshape the data to match the input shape of the LSTM model
    x_train = np.reshape(x_train.to_numpy(), (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test.to_numpy(), (x_test.shape[0], 1, x_test.shape[1]))
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    options = ['LSTM', 'GRU']
    # Create the option box using st.selectbox
    selected_option = st.sidebar.selectbox("Select model type:", options)
    model_type = selected_option

    st.sidebar.write("Lookback is the number of months for the model to consider when making predictions.")
    options = ['12', '24', '36', '48', '60', '72', '84', '96', '108', '120', '132', '144', '156', '168', '180', '192', '204', '216', '228', '240', '252', '264', '276']
    # Create the option box using st.selectbox
    selected_option = st.sidebar.selectbox("Set lookback:", options, index=9)
    look_back = int(selected_option)   

    n_features = 1  # Number of features in your typhoon data

    if model_type == 'LSTM':
        model = lstm_model(look_back, n_features)
 
    elif model_type == 'GRU':
        model = gru_model(look_back, n_features)
                    
    # Compile the model
    # Define a lower learning rate
    learning_rate = 0.001  # You can adjust this value as needed

    # Create an optimizer object with the desired learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model specifying the optimizer
    model.compile(loss="mse", optimizer=optimizer)

    # Print model summary
    model.summary()

    if st.sidebar.button("Start Training"):
        if "model" not in st.session_state:
            st.session_state.model = model
        progress_bar = st.progress(0, text="Training the prediction model, please wait...")           
        # Train the model
        history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_test, y_test))

        fig, ax = plt.subplots()  # Create a figure and an axes
        ax.plot(history.history['loss'], label='Train')  # Plot training loss on ax
        ax.plot(history.history['val_loss'], label='Validation')  # Plot validation loss on ax

        ax.set_title('Model loss')  # Set title on ax
        ax.set_ylabel('Loss')  # Set y-label on ax
        ax.set_xlabel('Epoch')  # Set x-label on ax

        ax.legend()  # Add legend
        st.pyplot(fig)
        st.session_state.model = model

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("LSTM Network training completed!") 

    years = st.sidebar.slider(   
        label="Number years to forecast:",
        min_value=1,
        max_value=6,
        value=4,
        step=1
    )

    if st.sidebar.button("Predictions"):
        if "model" not in st.session_state:
            st.error("Please train the model before making predictions.")  
            return
        
        # Get the predicted values and compute the accuracy metrics
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        st.write('Train RMSE:', train_rmse)
        st.write('Test RMSE:', test_rmse)
        st.write('Train MAE:', train_mae)
        st.write('Test MAE:', test_mae)

        model = st.session_state.model
        data_norm = st.session_state.data_norm
        # Get predicted data from the model using the normalized values
        predictions = model.predict(data_norm)

        # Inverse transform the predictions to get the original scale
        predvalues = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
  
        # Convert predvalues to integers
        predvalues_int = predvalues.astype(int)

        # Create a DataFrame with a column named "Typhoons"
        predvalues = pd.DataFrame({'Typhoons': predvalues_int.flatten()})

        # use the same index as the original data
        predvalues.index = df.index

        pred_period = years * 12    
        # Use the model to predict the next year of data
        input_seq_len = look_back         

        # check that look_back is less than the length of the data
        last_seq = data_norm[-input_seq_len:] 

        preds = []
        for i in range(pred_period):
            pred = model.predict(last_seq)
            preds.append(pred[0])

            last_seq = np.array(last_seq)
            last_seq = np.vstack((last_seq[1:], pred[0]))
            last_seq = pd.DataFrame(last_seq)

        # Inverse transform the predictions to get the original scale
        prednext = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        #flatten the array from 2-dim to 1-dim
        prednext = [item for sublist in prednext for item in sublist]

        # Convert prednext to integers
        prednext = [int(x) for x in prednext]

        end_dates = {
            12: '2024-12',
            24: '2025-12',
            36: '2026-12',
            48: '2027-12',
            60: '2028-12',
            72: '2029-12'
        }

        end = end_dates.get(pred_period, None)

        months = pd.date_range(start='2024-01', end=end, freq='MS')

        # Create a DataFrame with a column named "Typhoons"
        nextyear = pd.DataFrame(prednext, index=months, columns=["Typhoons"])

        # Concatenate along the rows (axis=0)
        combined_df = pd.concat([df, nextyear], axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Comparison of Actual and Predicted Number of Typhoons')

        ax.plot(predvalues['Typhoons'], color='red', linestyle='--', label='Model Predictions')  # dotted line for model predictions

        # Plot df's Typhoons values with one linestyle
        ax.plot(df['Typhoons'], color = 'blue', linestyle='-', label='Original Data')

        # Plot projected Typhoons values with a different linestyle
        ax.plot(combined_df.index[len(df):], combined_df['Typhoons'][len(df):], color = 'red', marker = 'o', linestyle='-', label='Projected Typhoons')

        max_y_value = max(df['Typhoons'].values.max(), nextyear['Typhoons'].max()) + 2
        ax.set_ylim(0, max_y_value)

        ax.set_xlabel('\nMonth', fontsize=20, fontweight='bold')
        ax.set_ylabel('Typhoons', fontsize=20, fontweight='bold')

        ax.set_xlabel('Month')
        ax.set_ylabel('Typhoons')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        # Add the legend
        ax.legend()  # This line adds the legend
        fig.tight_layout()
        st.pyplot(fig)

        st.write('Predicted Typhoons for the next', years, 'years:')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nextyear['Typhoons'], marker='o', linestyle='-')
        ax.set_title('Projected Number of Typhoons Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('No. of Typhoons')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Show Predicted Typhoons Table"):
            st.write("The Predicted Dataset")
            st.write(nextyear)   
            st.write(nextyear.shape)

        with st.expander("Read About the Lookback"):
            text = """When discussing the performance of an RNN LSTM (Recurrent Neural 
            Network Long Short-Term Memory) model on item Typhoons forecast data, it's 
            crucial to consider the impact of the "lookback" period on the model's 
            predictive capability. The "lookback" period refers to the number of 
            previous time steps the model uses to make predictions. In your scenario, 
            the lookback period is varied between 12 months and 36 months.
            \nWhen the lookback is set to a small number, such as 12 months, 
            the model is better at capturing short-term patterns and fluctuations 
            in Typhoons data. This is because it focuses on recent history, enabling it 
            to adapt quickly to changes in consumer behavior, seasonal trends, or 
            promotional activities. As a result, the model may exhibit strong 
            performance in predicting the changing patterns of Typhoons over shorter 
            time horizons.
            \nHowever, as the lookback period increases to 36 months or longer, 
            the model's ability to capture short-term fluctuations diminishes. 
            Instead, it starts to emphasize longer-term trends and patterns in the data. 
            While this may lead to more stable and consistent predictions over longer 
            time horizons, it may also result in a loss of sensitivity to short-term dynamics. 
            Consequently, the model may predict lower Typhoons overall, as it focuses more on 
            the average behavior observed over the extended period.
            \nIn essence, the choice of lookback period involves a trade-off 
            between capturing short-term fluctuations and maintaining a broader 
            perspective on long-term trends. A shorter lookback period enables the 
            model to react quickly to changes but may sacrifice accuracy over 
            longer periods. Conversely, a longer lookback period provides a more 
            stable outlook but may overlook short-term dynamics."""
            st.write(text)

def lstm_model(look_back, n_features):
    model =  tf.keras.Sequential([  
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(look_back, n_features)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64, return_sequences=True),  # Another GRU layer
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),  # Reduced units for final layer
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    return model

def gru_model(look_back, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True), input_shape=(look_back, n_features)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="tanh")
    ])
    return model

if __name__ == '__main__':
    app()   
