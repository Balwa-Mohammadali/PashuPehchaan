# import os
# import json
# import uuid
# from django.shortcuts import render, redirect
# from django.conf import settings
# from .forms import ImageUploadForm
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image

# # Lazy-loaded model and labels
# _model = None
# _labels = None

# def load_model_and_labels():
#     """Load Keras model and JSON labels once."""
#     global _model, _labels
#     if _model is None or _labels is None:
#         try:
#             model_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'Best.keras')
#             labels_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'Best.json')

#             if not os.path.exists(model_path) or not os.path.exists(labels_path):
#                 print("❌ Model or labels not found at:", model_path, labels_path)
#                 _model, _labels = None, None
#             else:
#                 _model = tf.keras.models.load_model(model_path, compile=False)
#                 with open(labels_path, 'r') as f:
#                     _labels = json.load(f)
#                 print("✅ Model and labels loaded successfully.")
#                 print("Model input shape:", _model.input_shape)
#                 print("Model output shape:", _model.output_shape)
#         except Exception as e:
#             _model, _labels = None, None
#             print("❌ Error loading model/labels:", e)
#     return _model, _labels

# def predict_cattle(img_path, model, class_names, alt_threshold=0.15):
#     """Predict breed from image file path."""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     preds = model.predict(img_array)[0]
#     top_indices = preds.argsort()[-2:][::-1]
#     top1, top2 = top_indices[0], top_indices[1]
#     conf1, conf2 = preds[top1], preds[top2]

#     if conf1 > 0.8:
#         confidence_label = "High Confidence"
#     elif conf1 > 0.6:
#         confidence_label = "Medium Confidence"
#     else:
#         confidence_label = "Low Confidence"

#     output = {
#         "prediction": class_names[top1],
#         "confidence": round(float(conf1) * 100, 2),
#         "confidence_label": confidence_label
#     }

#     if conf2 >= conf1 - alt_threshold:
#         output["second_opinion"] = {
#             "class": class_names[top2],
#             "confidence": round(float(conf2) * 100, 2)
#         }
#     return output

# # --- Views ---
# def home(request):
#     """Home page view."""
#     return render(request, 'classify/home.html')

# def about(request):
#     """About page view."""
#     return render(request, 'classify/about.html')

# def predict_page(request):
#     """Render the predict page with upload form."""
#     form = ImageUploadForm()
#     return render(request, 'classify/predict.html', {'form': form, 'preds': None, 'error': None, 'img_url': None})

# def predict(request):
#     """Handle POST request for breed prediction."""
#     error = None
#     preds = None
#     form = ImageUploadForm(request.POST or None, request.FILES or None)

#     if request.method == 'POST' and form.is_valid():
#         image_file = form.cleaned_data['image']

#         if not image_file:
#             error = "Uploaded file is empty. Please select a valid image."
#             return render(request, 'predict.html', {'form': form, 'preds': None, 'error': error, 'img_url': None})

#         # Save uploaded file
#         save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#         os.makedirs(save_dir, exist_ok=True)
#         unique_filename = f"{uuid.uuid4().hex}_{image_file.name}"
#         img_path = os.path.join(save_dir, unique_filename)
#         with open(img_path, 'wb+') as destination:
#             for chunk in image_file.chunks():
#                 destination.write(chunk)

#         # Generate URL for template preview
#         img_url = os.path.join(settings.MEDIA_URL, 'uploads', unique_filename)

#         # Load model and labels
#         model, labels = load_model_and_labels()
#         if model is None or not labels:
#             error = "Model or labels not found or failed to load."
#             preds = None
#         else:
#             try:
#                 preds = predict_cattle(img_path, model, labels)
#             except Exception as e:
#                 error = f"Prediction error: {e}"
#                 preds = None

#         return render(request, 'classify/predict.html', {'form': form, 'preds': preds, 'error': error, 'img_url': img_url})

#     return render(request, 'classify/predict.html', {'form': form, 'preds': preds, 'error': error, 'img_url': None})











import os
import json
import uuid
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Lazy-loaded model and labels
_model = None
_labels = None

def load_model_and_labels():
    """Load Keras model and JSON labels once."""
    global _model, _labels
    if _model is None or _labels is None:
        try:
            model_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'foreign_model.keras')
            labels_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'foreign_model.json')

            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                print("❌ Model or labels not found at:", model_path, labels_path)
                _model, _labels = None, None
            else:
                _model = tf.keras.models.load_model(model_path, compile=False)
                with open(labels_path, 'r') as f:
                    _labels = json.load(f)
                print("✅ Model and labels loaded successfully.")
                print("Model input shape:", _model.input_shape)
                print("Model output shape:", _model.output_shape)
        except Exception as e:
            _model, _labels = None, None
            print("❌ Error loading model/labels:", e)
    return _model, _labels

def preprocess_image(img_path):
    """Preprocess uploaded image to match RGB model input (129x129x3)."""
    img = image.load_img(img_path, target_size=(129, 129), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,129,129,3)
    return img_array

def predict_cattle(img_path, model, class_names, alt_threshold=0.15):
    """Predict breed from image path using RGB input."""
    try:
        img_array = preprocess_image(img_path)
        preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-2:][::-1]
        top1, top2 = top_indices[0], top_indices[1]
        conf1, conf2 = preds[top1], preds[top2]

        if conf1 > 0.8:
            confidence_label = "High Confidence"
        elif conf1 > 0.6:
            confidence_label = "Medium Confidence"
        else:
            confidence_label = "Low Confidence"

        output = {
            "prediction": class_names[top1],
            "confidence": round(float(conf1) * 100, 2),
            "confidence_label": confidence_label
        }

        if conf2 >= conf1 - alt_threshold:
            output["second_opinion"] = {
                "class": class_names[top2],
                "confidence": round(float(conf2) * 100, 2)
            }

        return output

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": f"Prediction failed: {str(e)}"}

# --- Views ---
def home(request):
    """Home page view."""
    return render(request, 'classify/home.html')

def about(request):
    """About page view."""
    return render(request, 'classify/about.html')

def predict_page(request):
    """Render the predict page with upload form."""
    form = ImageUploadForm()
    return render(request, 'classify/predict.html', {
        'form': form,
        'preds': None,
        'error': None,
        'img_url': None
    })

def predict(request):
    """Handle POST request for breed prediction."""
    error = None
    preds = None
    form = ImageUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        image_file = form.cleaned_data['image']

        if not image_file:
            error = "Uploaded file is empty. Please select a valid image."
            return render(request, 'classify/predict.html', {
                'form': form,
                'preds': None,
                'error': error,
                'img_url': None
            })

        # Save uploaded file
        save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(save_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4().hex}_{image_file.name}"
        img_path = os.path.join(save_dir, unique_filename)
        with open(img_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # Generate URL for template preview
        img_url = os.path.join(settings.MEDIA_URL, 'uploads', unique_filename)

        # Load model and labels
        model, labels = load_model_and_labels()
        if model is None or not labels:
            error = "Model or labels not found or failed to load."
            preds = None
        else:
            preds = predict_cattle(img_path, model, labels)
            if "error" in preds:
                error = preds["error"]
                preds = None

        return render(request, 'classify/predict.html', {
            'form': form,
            'preds': preds,
            'error': error,
            'img_url': img_url
        })

    return render(request, 'classify/predict.html', {
        'form': form,
        'preds': preds,
        'error': error,
        'img_url': None
    })









# import os
# import json
# import uuid
# from django.shortcuts import render
# from django.conf import settings
# from .forms import ImageUploadForm
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import tensorflow as tf

# # Lazy-loaded model and labels
# _model = None
# _labels = None


# def load_model_and_labels():
#     """Load Keras model and JSON labels once."""
#     global _model, _labels
#     if _model is None or _labels is None:
#         try:
#             # ✅ use the same names as you saved in Colab
#             model_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'Best.keras')
#             labels_path = os.path.join(settings.BASE_DIR, 'classify', 'ml', 'Best.json')

#             if not os.path.exists(model_path) or not os.path.exists(labels_path):
#                 print("❌ Model or labels not found at:", model_path, labels_path)
#                 _model, _labels = None, None
#             else:
#                 _model = tf.keras.models.load_model(model_path, compile=False)
#                 with open(labels_path, 'r') as f:
#                     _labels = json.load(f)
#                 print("✅ Model and labels loaded successfully.")
#                 print("Model input shape:", _model.input_shape)
#                 print("Model output shape:", _model.output_shape)
#         except Exception as e:
#             _model, _labels = None, None
#             print("❌ Error loading model/labels:", e)
#     return _model, _labels


# def preprocess_image(img_path):
#     """Preprocess uploaded image to match MobileNetV2 input (224x224x3)."""
#     img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
#     return img_array


# def predict_cattle(img_path, model, class_names, alt_threshold=0.15):
#     """Predict breed from image path using RGB input."""
#     try:
#         img_array = preprocess_image(img_path)
#         preds = model.predict(img_array)[0]

#         top_indices = preds.argsort()[-2:][::-1]
#         top1, top2 = top_indices[0], top_indices[1]
#         conf1, conf2 = preds[top1], preds[top2]

#         if conf1 > 0.8:
#             confidence_label = "High Confidence"
#         elif conf1 > 0.6:
#             confidence_label = "Medium Confidence"
#         else:
#             confidence_label = "Low Confidence"

#         output = {
#             "prediction": class_names[top1],
#             "confidence": round(float(conf1) * 100, 2),
#             "confidence_label": confidence_label
#         }

#         if conf2 >= conf1 - alt_threshold:
#             output["second_opinion"] = {
#                 "class": class_names[top2],
#                 "confidence": round(float(conf2) * 100, 2)
#             }

#         return output

#     except Exception as e:
#         print("❌ Prediction error:", e)
#         return {"error": f"Prediction failed: {str(e)}"}


# # --- Views ---
# def home(request):
#     """Home page view."""
#     return render(request, 'classify/home.html')


# def about(request):
#     """About page view."""
#     return render(request, 'classify/about.html')


# def predict_page(request):
#     """Render the predict page with upload form."""
#     form = ImageUploadForm()
#     return render(request, 'classify/predict.html', {
#         'form': form,
#         'preds': None,
#         'error': None,
#         'img_url': None
#     })


# def predict(request):
#     """Handle POST request for breed prediction."""
#     error = None
#     preds = None
#     form = ImageUploadForm(request.POST or None, request.FILES or None)

#     if request.method == 'POST' and form.is_valid():
#         image_file = form.cleaned_data['image']

#         if not image_file:
#             error = "Uploaded file is empty. Please select a valid image."
#             return render(request, 'classify/predict.html', {
#                 'form': form,
#                 'preds': None,
#                 'error': error,
#                 'img_url': None
#             })

#         # Save uploaded file
#         save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#         os.makedirs(save_dir, exist_ok=True)
#         unique_filename = f"{uuid.uuid4().hex}_{image_file.name}"
#         img_path = os.path.join(save_dir, unique_filename)
#         with open(img_path, 'wb+') as destination:
#             for chunk in image_file.chunks():
#                 destination.write(chunk)

#         # Generate URL for template preview
#         img_url = os.path.join(settings.MEDIA_URL, 'uploads', unique_filename)

#         # Load model and labels
#         model, labels = load_model_and_labels()
#         if model is None or not labels:
#             error = "Model or labels not found or failed to load."
#             preds = None
#         else:
#             preds = predict_cattle(img_path, model, labels)
#             if "error" in preds:
#                 error = preds["error"]
#                 preds = None

#         return render(request, 'classify/predict.html', {
#             'form': form,
#             'preds': preds,
#             'error': error,
#             'img_url': img_url
#         })

#     return render(request, 'classify/predict.html', {
#         'form': form,
#         'preds': preds,
#         'error': error,
#         'img_url': None
#     })
