import os
HEAD = 'baseline'  # change to 'baseline' or 'convgap' to run other variants

for i in range(3):
    cmd = f'python combinemodelkmu.py --model Ourmodel --bs 128 --lr 0.001 --fold {i+1} --head {HEAD}'
    os.system(cmd)

print("Train ShuffViT ok!")


# commands
# python confmatrixkmu.py --model Ourmodel --dataset efficientvitwcc --head mlp --start_fold 1 --end_fold 3
# python confmatrixkmu.py --model Ourmodel --dataset efficientvitwcc --head convgap --start_fold 1 --end_fold 3
# 0. organize_kmu_by_emotion.py
# 1. preprocess_kmu.py
# 2. KMU.py
# 3. 10fold.py
# 4. confirmatrixkmu.py
# 5. gradcam_kmu.py

# with mlp
# Avg per-image forward time (Train): 0.844762s | (Test): 0.682385s
# Epoch: 25 | Time: 0h 2m 44s
# Total Time: 1h 8m 32s | ~1.14 hours
# best_Test_acc: 96.364
# best_Test_acc_epoch: 22
# Train ShuffViT ok!

# with convogap
# Saving..  best_Test_acc: 94.545
# Avg per-image forward time (Train): 0.868777s | (Test): 0.707721s
# Epoch: 25 | Time: 0h 2m 43s
# Total Time: 1h 8m 23s | ~1.14 hours
# best_Test_acc: 94.545
# best_Test_acc_epoch: 24
# Train ShuffViT ok!