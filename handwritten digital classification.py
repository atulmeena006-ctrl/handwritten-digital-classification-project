import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import cv2
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.images
y = digits.target

processed_images = []

for image in X:
    image = cv2.normalize(
        image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    image = cv2.resize(image, (16, 16))
    processed_images.append(image.flatten())

X = np.array(processed_images)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Actual: {digits.target[i]}\nPredicted: {model.predict([X[i]])[0]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
