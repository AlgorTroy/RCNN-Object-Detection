from rcnn_model import BuildRCNN
from keras.datasets import cifar10

# load the dataset
(X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

print(X_train, y_train)
# build the model
model = BuildRCNN(nbChannels=3,
                  shape1=32,
                  shape2=32,
                  nbClasses=10,
                  nbRCL=5,
                  nbFilters=128,
                  filtersize=3)

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
model.fit(X_train, y_train, batch_size=64, nb_epoch=100, validation_data=(X_valid, y_valid))


