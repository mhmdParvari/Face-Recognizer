# Face Recognition
Identifying some iranian celebrities from their face image using TensorFlow and [deepface](https://github.com/serengil/deepface) framework.  
Check out `celebs.txt` to see which celebrities this program currently supports.

## Prerequisites
Run the following command to install the required libraries all at once:
```
pip install -r requirements.txt
```
This will install **NumPy**, **TensorFlow** and **deepface** on your computer.

## Usage
Prepare a face image and give it to the program using the command below:
```
python recognizer.py --input image_of_sb.jpg
```


## Performance

|          | Accuracy | Loss
| -------- | -------- | ---
Train      | 0.87     | 0.50
Validation |  0.80    | 0.72
