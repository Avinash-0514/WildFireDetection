
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

    def dense_or_default(sparse_tensor):
        dense = tf.sparse.to_dense(sparse_tensor, default_value=0.0)
        return dense[0] if tf.size(dense) == 1 else dense

    ndvi = dense_or_default(parsed_example['NDVI'])
    tmmn = dense_or_default(parsed_example['tmmn'])
    elevation = dense_or_default(parsed_example['elevation'])
    population = dense_or_default(parsed_example['population'])
    fire_mask = dense_or_default(parsed_example['FireMask'])
    vs = dense_or_default(parsed_example['vs'])
    pdsi = dense_or_default(parsed_example['pdsi'])
    pr = dense_or_default(parsed_example['pr'])
    tmmx = dense_or_default(parsed_example['tmmx'])
    sph = dense_or_default(parsed_example['sph'])
    th = dense_or_default(parsed_example['th'])
    prev_fire_mask = dense_or_default(parsed_example['PrevFireMask'])
    erc = dense_or_default(parsed_example['erc'])

    # You may want to reshape fire_mask and prev_fire_mask to (256,256) if they represent masks
    fire_mask = tf.reshape(fire_mask, [64, 64])
    prev_fire_mask = tf.reshape(prev_fire_mask, [64, 64])


    inputs = tf.stack([
        ndvi, tmmn, elevation, population, vs,
        pdsi, pr, tmmx, sph, th, erc
    ])

    return inputs, fire_mask, prev_fire_mask



# Create dataset pipeline
dataset = tf.data.TFRecordDataset(tfrecord_files)
dataset = dataset.map(_parse_function)
dataset = dataset.batch(4)  # small batch for demo
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Visualize some samples
for batch in dataset.take(1):
    inputs_batch, fire_masks_batch, prev_fire_masks_batch = batch
    print("Input features shape:", inputs_batch.shape)         # (batch_size, feature_count)
    print("FireMask shape:", fire_masks_batch.shape)           # (batch_size, 256, 256)
    print("PrevFireMask shape:", prev_fire_masks_batch.shape)  # (batch_size, 256, 256)
    
    # Plot first fire mask
    plt.figure(figsize=(8, 4))
    for i in range(inputs_batch.shape[0]):
        plt.subplot(1, inputs_batch.shape[0], i+1)
        plt.imshow(fire_masks_batch[i].numpy(), cmap='hot')
        plt.title(f'FireMask #{i}')
        plt.axis('off')
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