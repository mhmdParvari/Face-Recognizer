import argparse
import numpy as np
import tensorflow as tf
from deepface import DeepFace

def extract_features(img_path):
	embedding_objs = DeepFace.represent(img_path, model_name='ArcFace', enforce_detection=False)
	return embedding_objs[0]['embedding']

celebs = ['Ali_Daei', 'Ehsan_Alikhani', 'Elnaz_Shakerdoost', 'Adel_FerdowsiPour', 'Ali_Khamenei',
          'Asghar_Farhadi', 'Bahare_Rahnama', 'Bahram_Radan', 'Behnam_Bani', 'Dariush_Arjmand',
          'Elham_Hamidi', 'Golshifteh_Farahani', 'Hamid_Lolaei', 'Hootan_Shakiba', 'Javad_Khiabani',
          'Javad_Razavian', 'Leyla_Hatami', 'Mahnaz_Afshar', 'Mehran_Ghafourian', 'Mehran_Modiri',
          'Mohsen_Chavoshi', 'Parinaz_Izadyar', 'Parsa_Pirozfar', 'Parviz_Parastooee', 'Shahab_Hosseini',
          'Siamak_Ansari', 'Siavash_Ghomayshi', 'Tannaz_Tabatabaee', 'Taraneh_Alidoosti', 'Ebi']

def Recognize(img_path):
	model = tf.keras.models.load_model('weights/model.h5')
	feature_vector = extract_features(img_path)
	return celebs[np.argmax(model.predict([feature_vector]))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    celeb = Recognize(args.input)
    print(celeb)