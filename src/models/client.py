import requests
import json

endpoint = "http://20.76.229.140:80/api/v1/service/my-service/score"
x_new = ["Perfect for new parents. We were able to keep track of baby's feeding, sleep and diaper change schedule for the first two and a half months of her life. Made life easier when the doctor would ask questions about habits because we had it all right there!",
         "Perfect for new parents. We were able to keep track of baby's feeding, sleep and diaper change schedule for the first two and a half months of her life. Made life easier when the doctor would ask questions about habits because we had it all right there!"]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})
print(input_json)
# Set the content type

headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers=headers)
predictions = predictions.content.decode('ascii')
print(predictions)
#predicted_classes = json.loads(predictions.json())

#for i in range(len(x_new)):
#    print ("Patient {}".format(x_new[i]), predicted_classes[i] )
