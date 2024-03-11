# 2_3_flower17.py
import keras
import matplotlib.pyplot as plt

# flower17
#   - train
#      - buttercup
#      - coltsfoot
#      - daffodil
#   - test
#      - buttercup
#      - coltsfoot
#      - daffodil

gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    # zoom_range=(0.5, 2),
    width_shift_range=150,
)
flow = gen.flow_from_directory('flower17/train',
                               target_size=(224, 224),
                               class_mode='sparse',     # "categorical", "binary", "sparse"
                               batch_size=4,
                               shuffle=False
                               )

# for batch in flow:
#     print(len(batch))       # 2
#     break

for x, y in flow:
    print(x.shape, y.shape)     # (4, 224, 224, 3) (4,)
    print(y)                    # [1. 0. 0. 0.]

    plt.imshow(x[0])
    break

plt.show()

# print(flow.classes)
# print(flow.labels)
print(flow.class_indices)
