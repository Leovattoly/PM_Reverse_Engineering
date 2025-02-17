# Predicting Optimal Particle Morphology Conditions: A Reverse Engineering Approach:


---

##  Repository Structure

──  Data/ # Contains datasets for training, testing & optimization <br>
      &nbsp;  &nbsp;│──  train.csv # Training the AE model <br>
      &nbsp;  &nbsp;│──  test.csv # Training the LSTM model<br>
      &nbsp;  &nbsp;│── test_test_dataset.csv # Testing both AE & LSTM models<br>
      &nbsp;  &nbsp;│── opt.csv # used in the GA to verify the model efficiency<br>
      **Unable to upload train.csv & test.csv due to size<br>**
      
│──  Dataset.py # POLYMAT simulator for creating datasets<br>
│──  Autoencoder.py # Defines & trains the Autoencoder model<br>
│──  decoder_model_16500.sav # Pretrained decoder model<br>
│──  encoder_model_16500.sav # Pretrained encoder model<br>
│──  LSTM_Input_dataset.py # Prepares input data for LSTM model training<br>
│──  LSTM_target_dataset_maker.py # Creates target datasets for LSTM training<br>
│──  LSTM_model.py # Defines & trains the LSTM model<br>
│──  LSTM_model.sav # Pretrained LSTM model<br>
│──  GA_customized.py # Implementation of Genetic Algorithm for optimization<br>
│──  optimization_dataset.py # Dataset creation script for optimization<br>



