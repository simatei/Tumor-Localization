class SavedModel(object):

    def __init__(self, model_def_path, model_parameters_path):
        self.session = tf.Session()
        self.graph = self.session.graph

        new_saver = tf.train.import_meta_graph(model_def_path)
        new_saver.restore(self.session, model_parameters_path)

        self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self.logits = self.graph.get_tensor_by_name("conv_net_1/logits:0")
        self.features = self.graph.get_tensor_by_name("conv_net_1/features:0")
        self.images = self.graph.get_tensor_by_name("conv_net_1/images:0")
        self.outputs = tf.nn.softmax(self.logits)

    def predict(self, feed_images, feed_features):
        feed_dict = {
            self.keep_prob: 1.0,
            self.images: feed_images,
            self.features: feed_features
        }
        return self.session.run(self.outputs, feed_dict=feed_dict)
