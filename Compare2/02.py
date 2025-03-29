import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# 参数设置
img_width, img_height = 128, 128
batch_size = 32
epochs = 20
data_dir = 'Pic2'  # 修改为你的数据路径

# 数据生成器
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80%训练，20%验证/测试
    rotation_range=20,     # 数据增强参数
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 训练数据生成器
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 验证/测试数据生成器
test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 构建轻量级CNN模型
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# 评估模型
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算分类报告
report = classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys()), output_dict=True)
precision = report['macro avg']['precision']
recall = report['macro avg']['recall']
f1 = report['macro avg']['f1-score']

# 计算特异性
cm = confusion_matrix(y_true, y_pred_classes)
specificities = []
for i in range(cm.shape[0]):
    tn = cm[np.arange(cm.shape[0]) != i, :][:, np.arange(cm.shape[0]) != i].sum()
    fp = cm[:, i].sum() - cm[i, i]
    specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    specificities.append(specificity_i)
avg_specificity = np.mean(specificities)

# 输出结果
print(f"\n{' Metric ':=^40}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {avg_specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print('='*40 + '\n')

# 输出分类详情
print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))