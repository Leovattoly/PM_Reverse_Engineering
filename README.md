# Predicting Optimal Particle Morphology Conditions: A Reverse Engineering Approach:


---

##  Repository Structure

──  Data/- **Contains datasets for training, testing & optimization** <br>
      &nbsp;  &nbsp;│──  train.csv &nbsp; Training the AE model <br>
      &nbsp;  &nbsp;│──  test.csv &nbsp; Training the LSTM model<br>
      &nbsp;  &nbsp;│── test_test_dataset.csv &nbsp; Testing both AE & LSTM models<br>
      &nbsp;  &nbsp;│── opt.csv &nbsp; used in the GA to verify the model efficiency<br>
      **Unable to upload train.csv & test.csv due to size<br>**
      
│──  **Dataset.py** &nbsp; POLYMAT simulator for creating datasets<br>
│──  **Autoencoder.py** &nbsp; Defines & trains the Autoencoder model<br>
│──  **decoder_model_16500.sav** &nbsp; Pretrained decoder model<br>
│──  **encoder_model_16500.sav** &nbsp; Pretrained encoder model<br>
│──  **LSTM_Input_dataset.py** &nbsp; Prepares input data for LSTM model training<br>
│──  **LSTM_target_dataset_maker.py** &nbsp; Creates target datasets for LSTM training<br>
│──  **LSTM_model.py** &nbsp; Defines & trains the LSTM model<br>
│──  **LSTM_model.sav** &nbsp; Pretrained LSTM model<br>
│──  **GA_customized.py** &nbsp; Implementation of Genetic Algorithm for optimization<br>
│──  **optimization_dataset.py** &nbsp; Dataset creation script for optimization<br>



