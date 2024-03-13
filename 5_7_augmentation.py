# 5_7_augmentation.py
import Augmentor

p = Augmentor.Pipeline('aug', '../new')

p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.8)
p.rotate90(probability=0.1)

p.sample(10)
