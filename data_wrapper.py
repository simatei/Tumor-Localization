import tensorflow as tf
import os


class DataReader(object):

    def get_train_batch(self, batch_size):
        return self.get_batch(batch_size, 'images/train/')

    def get_test_batch(self, batch_size):
        return self.get_batch(batch_size, 'images/test/')

    def get_batch(self, batch_size, image_dir):
        input_files = [image_dir + fname for fname in os.listdir(image_dir)
                       if not fname.startswith('.')]
        assert all(os.path.isfile(f) for f in input_files)
        queue = tf.train.string_input_producer(input_files, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([1], dtype=tf.int64),
                'image_raw': tf.FixedLenFeature([1], dtype=tf.string),
                'features': tf.FixedLenFeature([4], dtype=tf.int64)
            }
        )

        image = features['image_raw']
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [64, 64, 1])
        image = (tf.cast(image, tf.float32) / 255.0) - 0.5

        tensors = [
            features['label'],
            image,
            features['features']
        ]

        labels, images, features = tf.train.shuffle_batch(
            tensors=tensors,
            batch_size=batch_size,
            capacity=2000,
            min_after_dequeue=500,
            num_threads=6
        )

        images = tf.reshape(images, [-1, 64, 64, 1])
        labels = tf.squeeze(labels)
        features = tf.reshape(features, [-1, 4])
        features = tf.cast(features, tf.float32)

        return labels, images, features
