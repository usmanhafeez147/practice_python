# import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import numpy as np

# def predict_objects_in_image(image_path):
#     # Load the pre-trained VGG16 model
#     model = VGG16(weights='imagenet')

#     # Load and preprocess the image
#     img = image.load_img(image_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     # Make predictions
#     predictions = model.predict(x)

#     # Decode the predictions
#     decoded_predictions = decode_predictions(predictions, top=5)[0]

#     # Print the top 5 predictions
#     for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#         print(f"{i + 1}: {label} ({score:.2f})")

# # Example usage
# image_path = './test.jpg'
# predict_objects_in_image(image_path)


# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import numpy as np

# def detect_tongue(image_path):
#     # Load pre-trained MobileNetV2 model (you might need a more specialized model for your use case)
#     model = MobileNetV2(weights='imagenet')

#     # Load and preprocess the image
#     img = image.load_img(image_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     # Make predictions
#     predictions = model.predict(x)

#     # Decode the predictions
#     decoded_predictions = decode_predictions(predictions, top=5)[0]

#     # Print the top predictions
#     for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#         print(f"{i + 1}: {label} ({score:.2f})")

# # Example usage
# image_path = 'test.jpg'
# detect_tongue(image_path)



import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def detect_tongue(image_path):
    # Load pre-trained MobileNetV2 model (you might need a more specialized model for your use case)
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions = model.predict(x)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")
    
    # print(f"{i + 1}: {label} ({score:.2f})")
    # Check if "tongue" is in the top predictions
    # has_tongue = False
    # for _, label, _ in decoded_predictions:
    #     if "tongue" in label.lower():
    #         has_tongue = True
    #         break

    # if has_tongue:
    #     print("Tongue detected in the image.")
    # else:
    #     print("No clear image of a tongue found in the image.")

# Example usage
image_path = 'tongue.jpeg'
detect_tongue(image_path)

