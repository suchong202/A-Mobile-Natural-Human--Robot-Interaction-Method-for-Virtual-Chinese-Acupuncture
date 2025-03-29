import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


class GatingMechanism(Layer):
    def __init__(self, units, **kwargs):
        super(GatingMechanism, self).__init__(**kwargs)
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(units, activation='relu')
        self.gate = Dense(units, activation='sigmoid')

    def call(self, inputs):
        # Assume inputs is a list of two tensors [feature1, feature2]
        feature1, feature2 = inputs

        # Map features to the same dimension
        transformed_feature1 = self.dense1(feature1)
        transformed_feature2 = self.dense2(feature2)

        # Gate mechanism
        gate = self.gate(tf.concat([transformed_feature1, transformed_feature2], axis=-1))

        # Apply gate to each feature
        gated_feature1 = gate * transformed_feature1
        gated_feature2 = gate * transformed_feature2

        # Fuse features by summing
        fused_feature = gated_feature1 + gated_feature2

        return fused_feature


def gate(A,B):



    L1 = np.array(A)
    L2 = np.array(B)

    input_dim1 = len(L1[0])
    input_dim2 = len(L2[0])

    output_dim = len(L1[0])  # Desired output dimension after fusion
    # Create input tensors
    input1 = Input(shape=(input_dim1,))
    input2 = Input(shape=(input_dim2,))

    # Create GatingMechanism layer
    gating_layer = GatingMechanism(units=output_dim)

    # Apply GatingMechanism to the input features
    fused_feature = gating_layer([input1, input2])

    # Create a simple model
    model = Model(inputs=[input1, input2], outputs=fused_feature)

    # Compile the model (for demonstration purposes, we use a dummy loss function)
    model.compile(optimizer='adam', loss='mse')

    # Print model summary
    model.summary()
    # Predict using the model
    predictions = model.predict([L1,L2])

    return predictions

if __name__ == '__main__':
    # Example usage
    # Define dimensions of the input features
    input_dim1 = 9  # Dimension of the first feature vector
    input_dim2 = 9  # Dimension of the second feature vector

    # Generate some random data for testing
    data1 = np.random.rand(1, input_dim1)  # Batch of 5 samples, each with input_dim1 features
    data2 = np.random.rand(1, input_dim2)  # Batch of 5 samples, each with input_dim2 features



    print(data1)
    print(data2)

    print(gate(data1,data2))
