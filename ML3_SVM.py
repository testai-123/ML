import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from skimage.transform import resize

# Load the digits dataset
digits = datasets.load_digits()

# Split data into features (X) and labels (y)
X, y = digits.data, digits.target

# Display the 9th image (resize for larger display)
image_resized = resize(X[8].reshape(8, 8), (28, 28), anti_aliasing=True)
plt.imshow(image_resized, cmap='gray')
plt.title("Sample Digit Image")
plt.show()

# Plot the count of each digit
sns.countplot(x=y)
plt.xlabel('Digit Label')
plt.ylabel('Count')
plt.title('Count of Each Digit in the Dataset')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Display accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize some test images with predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    plt.axis('off')
plt.show()


# Get Index from the User
seq = int(input("Enter the Index : "))
plt.imshow(X_test[seq].reshape(8, 8), cmap='gray')
plt.title(f"Predicted : {y_pred[seq]}, Actual : {y_test[seq]}")
plt.axis('on')
