import requests
import json

endpoint = "http://2ec89b4e-dab7-4463-bf81-85911867487b.northeurope.azurecontainer.io/score"
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})
print(input_json)
# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
predictions = predictions.content.decode('ascii')
print(predictions)
#predicted_classes = json.loads(predictions.json())

#for i in range(len(x_new)):
#    print ("Patient {}".format(x_new[i]), predicted_classes[i] )