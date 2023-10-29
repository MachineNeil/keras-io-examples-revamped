"""
Created on: October 29, 2023.

Based on [Mark Doust](https://github.com/MarkDaoust)'s [creation](https://github.com/keras-team/keras-io/blob/bdb1a19d989668fc4c0ca09c572d4c8f6fe5c6b6/examples/structured_data/imbalanced_classification.py).

The [dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud/) belongs to Kaggle.

All the code below is my own and I allow its use and reproduction, as well as encourage its improvement both in situ and ex situ and hope it will be useful to someone.

TP: true positive - deemed legitimate and is legitimate (1, 1).
FP: false positive - deemed legitimate and is fraudulent (1, 0).
FN: false negative - deemed fraudulent and is legitimate (0, 1).
TN: true negative - deemed fraudulent and is fraudulent (0, 0).

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


'''
### Constants:

'''
# For reproducibility.
random_state = 42

# Since there are many more TPs than TNs, having an FN should be penalized more.
rel_importance = 50
batch_size = 2048

# Andrej Karpathy dixit - it seems to work as well with Nadam.
ini_lr = 3e-4
steps = 250
decay = 0.7
patience = 6
epochs = 60


def savefig(kind):
    # This function creates explicit file names, although there must be a more proper way to be orderly.
    plt.savefig('{}--loss-{:.4f}_acc-{:.4f}_relimp-{}_steps-{}_decay-{}_epochs-{}.png'.format(
        kind, loss, accuracy, rel_importance, steps, decay, epochs))


'''
### Prepare the data before training:

'''
# Read the csv database and create a dataframe with pandas.
df = pd.read_csv('./creditcard.csv')

# Get rid of the last column (targets). I choose not to specify the name of the column in case the database's creator modifies it in the future. x will be our features.
x = df.drop(df.columns[-1], axis=1)

# Get the column we have just dropped and turn it into a one-hot encoded vector.
y = to_categorical(df[df.columns[-1]], num_classes=2)

print('Shape of the features and targets:', x.shape, y.shape, '\n')


'''
### Preparing a training, testing, and validation set:

'''
# This was Mark Doust's. I believe it can be improved upon.
test_proportion = int(len(x) * 0.2)

# Obtain a 80-10-10 split between training, validation, and testing.
x, x_test, y, y_test = train_test_split(
    x, y,
    test_size=test_proportion,
    random_state=random_state
)
x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=test_proportion,
    random_state=random_state
)

print('{} training, {} testing, and {} validation samples.\n'.format(
    len(x_train), len(x_test), len(x_val)))


'''
### Analyzing class imbalance in the targets:

'''
# How many ones (fraudulent case) there are in the training set.
pos_examples = int(np.sum(y_train[:, 1]))

# How many zeros (legitimate case) there are in the training set.
neg_examples = len(y_train) - pos_examples

print('Positive samples in the training data: {} ({:.2f}% of total)'.format(
    pos_examples, 100 * float(pos_examples) / len(y_train)), '\n'
)


'''
### Normalizing the data:

'''
# Data normalization (substracting mu, dividing by sigma).
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


'''
### Building a binary classification model:

'''
model = Sequential([
    Input(
        shape=x_train.shape[-1]
    ),
    Dense(
        units=512,
        kernel_initializer='he_normal',
        use_bias=False
    ),
    BatchNormalization(),
    Activation('elu'),
    Dense(
        units=256,
        kernel_initializer='he_normal',
        use_bias=False
    ),
    BatchNormalization(),
    Activation('elu'),
    Dropout(0.3),
    Dense(
        units=256,
        kernel_initializer='he_normal',
        use_bias=False
    ),
    BatchNormalization(),
    Activation('elu'),
    Dropout(0.3),
    Dense(
        units=2,
        activation='sigmoid'
    )
])


'''
### Training the model considering the class imbalance:
 
'''
learning_rate = ExponentialDecay(
    initial_learning_rate=ini_lr,
    decay_steps=steps,
    decay_rate=decay
)

optimizer = Nadam(
    learning_rate=learning_rate
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        filepath='fraud_model.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,

        # This clarifies the monitored parameter must must decrease.
        mode='min'
    ),
    EarlyStopping(
        monitor='loss',
        min_delta=5e-6,
        patience=patience,
        verbose=1,
        mode='min',

        # Warm-up period.
        start_from_epoch=5
    )
]

# While 0 (non-fraudulent) has a weight of 1, 1 (fraudulent) has a weight of rel_importance, for which I found 50 works the best.
class_weight = {
    0: 1,
    1: rel_importance
}

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=False,
    callbacks=callbacks,
    validation_data=(x_val, y_val),
    class_weight=class_weight
)


'''
### Plotting the results

'''
model.summary()

loss, accuracy = model.evaluate(
    x_test, y_test,
    batch_size=batch_size,
    verbose=False
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
savefig('graph')
plt.show()


predicted_value = []
test = []
for i in model.predict(x_test):
    predicted_value.append(np.argmax(i))

for i in y_test:
    test.append(np.argmax(i))

plt.figure(figsize=(10, 10))
cm = confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True, fmt='d')
savefig('matrix')
plt.show()


'''
### Conclusions:

Considering a FN to be 100x more relevant than a FP, given 56,961 validation transactions:
- 10 out of 98 fraudulent transactions are misclassified as legitimate (FN) - very dangerous.
- 196 out of 56,863 valid transactions are misclassified as illegitimate (FP) - harmless, yet nagging.

Raising the weight of FNs lowers their occurrence, but in turn FPs soar. This effect has been reduced compared to the original creator's version, although minimizing it has proved to be an insurmountable task.

'''
