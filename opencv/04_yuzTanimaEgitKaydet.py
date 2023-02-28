import cv2, os
import numpy as np

# Eğitim verilerinin bulunduğu klasör yolu
dataset_path = 'opencv/data/train_dataset/'
train_saved_path = dataset_path+"trained_model_lbph.xml"
names_saved_path = dataset_path+"trained_model_lbph_names.txt"

# Yuz Algilama Modeli
face_cascade = cv2.CascadeClassifier('opencv/data\haarcascades\haarcascade_frontalface_default.xml')

# Tüm resimleri yüzleri etiketleriyle eğitin
def trainSave(dataset_path):
    names, faces, labels, label = [],[],[],0

    for root, dirs, files in os.walk(dataset_path):
        name = os.path.basename(root)
        label +=1
        
        print(f"\n\nFace '{name}' detecting.\n\tPictures: ",end="")

        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces_detector = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5);
            
            if (len(faces_detector) == 0):
                break
            
            img_name = os.path.basename(img_path)
            print(img_name,end=", ")

            (x, y, w, h) = faces_detector[0]
            detected = img[y:y+w, x:x+h]
            
            faces.append(detected)
            labels.append(label)
            names.append(name)

    # Yüz tanıma modeli LBPH oluşturma
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Traning starterd...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(train_saved_path)
    print(f"Traning successfully, closed and saved ({train_saved_path}) !")
    print(f"\nTraning names ({names}) is saving...")
    np.savetxt(names_saved_path, np.array(names),fmt="%s")
    print(f"\nSuccessfully names is ({names_saved_path}) saved!")

trainSave(dataset_path)
