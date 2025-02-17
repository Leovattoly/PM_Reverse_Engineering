# Predicting Optimal Particle Morphology Conditions: A Reverse Engineering Approach:


---

##  Repository Structure

──  Data/ # Contains datasets for training, testing & optimization
      │──  train.csv # Training the AE model 
      │──  test.csv # Training the LSTM model
      │── test_test_dataset.csv # Testing both AE & LSTM models
      │── opt.csv # used in the GA to verify the model efficiency
      ## ** Unable to upload train.csv & test.csv due to size
      
│──  Dataset.py # POLYMAT simulator for creating datasets
│──  Autoencoder.py # Defines & trains the Autoencoder model
│──  decoder_model_16500.sav # Pretrained decoder model
│──  encoder_model_16500.sav # Pretrained encoder model
│──  LSTM_Input_dataset.py # Prepares input data for LSTM model training
│──  LSTM_target_dataset_maker.py # Creates target datasets for LSTM training
│──  LSTM_model.py # Defines & trains the LSTM model
│──  LSTM_model.sav # Pretrained LSTM model
│──  GA_customized.py # Implementation of Genetic Algorithm for optimization
│──  optimization_dataset.py # Dataset creation script for optimization



