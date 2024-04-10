from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2 as cv
import numpy as np
import recognizer

app = FastAPI()

@app.post('/')
def recognize_face(input: UploadFile = File(None)):
    if not input.content_type.startswith('image/'):
        raise HTTPException(415, 'File type not supported')

    content = input.read()
    np_arr = np.frombuffer(content)
    img = cv.imdecode(np_arr, cv.IMREAD_UNCHANGED)
    cv.imwrite('image.jpg', img)
    celeb = recognizer.Recognize('image.jpg')
    return {'celebrity': celeb}