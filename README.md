# Face Recognition
Identifying some Iranian celebrities from their face image using TensorFlow and [deepface](https://github.com/serengil/deepface) framework.  
Check out `celebs.txt` to see which celebrities this program currently supports.

## Prerequisites
Run the following command to install the required base libraries:
```
pip install -r requirements.txt
```
This will install **NumPy**, **TensorFlow** and **deepface** on your computer.

## Usage
You can use this program either on its own (standalone) or through API.
### Standalone
Prepare a face image and give it to the program using the command below:
```
python recognizer.py --input image_of_sb.jpg
```
### API
First run this command to install the API related packages.
```
pip install -r api_requirements.txt
```
Then run this command to start the server locally:
```
uvicorn main:app
```
At last, prepare your image and send it to the API using an app like Postman.

## Performance

|          | Accuracy | Loss
| -------- | -------- | ---
Train      | 0.87     | 0.50
Validation |  0.80    | 0.72
