import numpy as np
import pickle

 
loaded_model = pickle.load(open('C:/Users/aasho/OneDrive/Desktop/deploymodel/logreg.pkl','rb'))


input_data = (842, 2.2, 1, 7, 0.6, 188, 2, 2, 20, 756, 2549, 9, 7, 19)

#changing the input data as numpy array
input_data_as_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_array.reshape(1, -1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('lower end phone')

elif prediction[0] == 1:
    print("little intermediate end phone")

elif prediction[0] == 2:
    print("intermediate end phone")

else:
    print("very lower end phone")
