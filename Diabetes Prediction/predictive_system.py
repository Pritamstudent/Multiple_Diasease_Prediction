import pickle
import numpy as np
loaded_model = pickle.load(open('trained_model.sav','rb'))
input_data = (0,137,40,35,168,43.1,2.288,33)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array to one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


#do prediction
prediction = loaded_model.predict(input_data_reshaped )

if(prediction[0] == 0):
  print("The person does not have diabetes")
else:
  print("The person has diabetes")