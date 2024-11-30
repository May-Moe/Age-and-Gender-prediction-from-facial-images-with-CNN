import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications import VGG16
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def load_dataset(path, limit=None):
    pixels = []
    age = []
    gender = []
    for i, img_name in enumerate(os.listdir(path)):
        if limit and i >= limit:
            break
        try:
            ages = int(img_name.split("_")[0])
            genders = int(img_name.split("_")[1])
            
            if genders not in [0, 1]:
                print(f"Skipping image {img_name} due to invalid gender label: {genders}")
                continue
                
            img = cv2.imread(os.path.join(path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            pixels.append(np.array(img))
            age.append(ages)
            gender.append(genders)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue
    
    return np.array(pixels), np.array(age, dtype=np.int64), np.array(gender, dtype=np.uint64)

def train_and_evaluate(pixels, age, gender, model_name):
    x_train, x_test, y_train, y_test = train_test_split(pixels, age, random_state=100)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(pixels, gender, random_state=100)

    input = Input(shape=(200, 200, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input)
    x = base_model.output
    x = Flatten()(x)

    # Age prediction branch
    age_l = Dense(128, activation="relu")(x)
    age_l = Dense(64, activation="relu")(age_l)
    age_l = Dense(32, activation="relu")(age_l)
    age_l = Dense(1, activation="linear")(age_l)

    # Gender prediction branch
    gender_l = Dense(128, activation="relu")(x)
    gender_l = Dense(64, activation="relu")(gender_l)
    gender_l = Dense(32, activation="relu")(gender_l)
    gender_l = Dropout(0.5)(gender_l)
    gender_l = Dense(2, activation="softmax")(gender_l)

    model = Model(inputs=base_model.input, outputs=[age_l, gender_l])

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="adam", loss=["mse", "sparse_categorical_crossentropy"], metrics=['mae', 'accuracy'])

    # Train the model
    history = model.fit(x_train, [y_train, y_train_2], validation_data=(x_test, [y_test, y_test_2]), epochs=10)

    # Evaluate the model
    print(f"Evaluating the model trained on {model_name} dataset...")

    # Gender prediction evaluation
    y_pred_gender = np.argmax(model.predict(x_test)[1], axis=1)
    print("Gender Prediction Evaluation")
    print(confusion_matrix(y_test_2, y_pred_gender))
    print(classification_report(y_test_2, y_pred_gender))

    # Age prediction evaluation
    y_pred_age = model.predict(x_test)[0]
    print("Age Prediction Evaluation")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_age))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_age))

    # Save results
    np.save(f"{model_name}_true_age.npy", y_test)
    np.save(f"{model_name}_pred_age.npy", y_pred_age)
    np.save(f"{model_name}_true_gender.npy", y_test_2)
    np.save(f"{model_name}_pred_gender.npy", y_pred_gender)

    model.save(f"{model_name}_model.keras")

# Load UTKFace dataset
utk_path = "UTKFace"
utk_pixels, utk_age, utk_gender = load_dataset(utk_path, limit=1000)

# Load your own small dataset
your_path = "OwnFace"
your_pixels, your_age, your_gender = load_dataset(your_path, limit=35)

# Train and evaluate on UTKFace dataset
train_and_evaluate(utk_pixels, utk_age, utk_gender, "UTKFace")

# Train and evaluate on your own small dataset
train_and_evaluate(your_pixels, your_age, your_gender, "OwnFace")
