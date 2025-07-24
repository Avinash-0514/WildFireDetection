
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Path where your extracted TFRecord files are located
tfrecord_folder = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/Applied Project/archive"

# List all tfrecord files in folder
tfrecord_files = [os.path.join(tfrecord_folder, f) for f in os.listdir(tfrecord_folder) if f.endswith('.tfrecord')]
print(f"Found {len(tfrecord_files)} TFRecord files.")

# Parsing function for each example
def _parse_function(example_proto):
    feature_description = {
        "NDVI": tf.io.VarLenFeature(tf.float32),
        "tmmn": tf.io.VarLenFeature(tf.float32),
        "elevation": tf.io.VarLenFeature(tf.float32),
        "population": tf.io.VarLenFeature(tf.float32),
        "FireMask": tf.io.VarLenFeature(tf.float32),
        "vs": tf.io.VarLenFeature(tf.float32),
        "pdsi": tf.io.VarLenFeature(tf.float32),
        "pr": tf.io.VarLenFeature(tf.float32),
        "tmmx": tf.io.VarLenFeature(tf.float32),
        "sph": tf.io.VarLenFeature(tf.float32),
        "th": tf.io.VarLenFeature(tf.float32),
        "PrevFireMask": tf.io.VarLenFeature(tf.float32),
        "erc": tf.io.VarLenFeature(tf.float32),
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    def reshape(sparse_tensor):
        dense = tf.sparse.to_dense(sparse_tensor, default_value=0.0)
        return tf.reshape(dense, [64, 64])

    ndvi = reshape(parsed_example['NDVI'])
    tmmn = reshape(parsed_example['tmmn'])
    elevation = reshape(parsed_example['elevation'])
    population = reshape(parsed_example['population'])
    fire_mask = reshape(parsed_example['FireMask'])
    vs = reshape(parsed_example['vs'])
    pdsi = reshape(parsed_example['pdsi'])
    pr = reshape(parsed_example['pr'])
    tmmx = reshape(parsed_example['tmmx'])
    sph = reshape(parsed_example['sph'])
    th = reshape(parsed_example['th'])
    prev_fire_mask = reshape(parsed_example['PrevFireMask'])
    erc = reshape(parsed_example['erc'])

    # Stack into tensor for modeling
    inputs = tf.stack([
        ndvi, tmmn, elevation, population, vs,
        pdsi, pr, tmmx, sph, th, erc
    ], axis=0)  # Shape: (11, 64, 64)

    return inputs, fire_mask, prev_fire_mask


# Create dataset pipeline
dataset = tf.data.TFRecordDataset(tfrecord_files)
dataset = dataset.map(_parse_function)
dataset = dataset.batch(4)  # small batch for demo
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Visualize some samples
for batch in dataset.take(1):
    inputs_batch, fire_masks_batch, prev_fire_masks_batch = batch

    print("Inputs batch shape:", inputs_batch.shape)  # (B, 11, 64, 64)
    print("FireMask shape:", fire_masks_batch.shape)  # (B, 64, 64)

    # Plot the input features for the first sample in batch
    sample_inputs = inputs_batch[0]  # shape: (11, 64, 64)
    feature_names = [
        "NDVI", "tmmn", "elevation", "population", "vs",
        "pdsi", "pr", "tmmx", "sph", "th", "erc"
    ]

    plt.figure(figsize=(15, 6))
    for i in range(11):
        plt.subplot(2, 6, i + 1)
        plt.imshow(sample_inputs[i].numpy(), cmap='viridis')
        plt.title(feature_names[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Also show FireMask and PrevFireMask
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(fire_masks_batch[0].numpy(), cmap='hot')
    plt.title("FireMask")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(prev_fire_masks_batch[0].numpy(), cmap='hot')
    plt.title("PrevFireMask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


'''
import tensorflow as tf
import os

TFRECORD_DIR = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/Applied Project/archive"
tfrecord_files = [os.path.join(TFRECORD_DIR, f) for f in os.listdir(TFRECORD_DIR)
                  if f.endswith('.tfrecord') or f.endswith('.tfrecords')]

raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

for i, raw_record in enumerate(raw_dataset.take(10)):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(f"\nRecord #{i} feature keys:")
    for key in example.features.feature.keys():
        print(f" - {key}")
        '''
'''
import tensorflow as tf
import os

tfrecord_dir = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/Applied Project/archive"  # or wherever you extracted the files
tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith(".tfrecord")]

total_count = 0
for tfrecord_file in tfrecord_files:
    for _ in tf.data.TFRecordDataset(tfrecord_file):
        total_count += 1

print(f"Total number of samples: {total_count}")
'''