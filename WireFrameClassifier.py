 
import cv2  
import matplotlib.pyplot as plt       
import numpy as np     
from tensorflow.keras.models import load_model         
         
# Load the trained classification model              
model = load_model('CNN.keras')            
             
# Load the image                 
image = cv2.imread('../images/mo4.jpg')                   
classes = ['button','imagecercle','imagerectangle','input','logo','text']                 
# Convertir l'image en niveaux de gris             
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             
           
# Appliquer un flou gaussien pour réduire le bruit              
blurred = cv2.GaussianBlur(gray, (5, 5), 0)             
          
# Detecting edges with Canny's algorithm              
edged = cv2.Canny(blurred, 30, 150)            
        
# Trouver les contours dans l'image            
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)           
        
# Pour chaque contour détecté            
for contour in contours:              
    # Extract coordinates from bounding box             
    x, y, w, h = cv2.boundingRect(contour)           
        
    # Extraire la région d'intérêt (ROI) de l'image originale        
    roi = image[y:y+h, x:x+w]       
       
    # Redimensionner la ROI en 224x224     
    roi_resized = cv2.resize(roi, (224, 224))   
 
    # Prétraiter la ROI pour l'adapter à l'entrée du modèle  
    roi_preprocessed = np.expand_dims(roi_resized, axis=0)  # Ajouter une dimension batch  
    roi_preprocessed = roi_preprocessed / 255.0  # Normaliser les valeurs des pixels  
     
    # Appliquer votre modèle de classification à la ROI prétraitée 
    prediction = model.predict(roi_preprocessed) 
    predicted_class = np.argmax(prediction) 
    confidence = prediction[0][predicted_class] 
         
    # If confidence is high enough, draw bounding box and label 
     
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 
    cv2.putText(image, str(classes[predicted_class]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
 
# Afficher l'image avec les rectangles autour des ROIs  
cv2.imshow('Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
