import tensorflow as tf


class MetricsManager:

    def __init__(self):
        self.metrics = {
            "GDICEL": self.generalize_dice_loss,
            "DICEL": self.dice_loss,
            "L1": self.L1_loss,
            "L2": self.L2_loss,
            "L1_summed": self.L1_loss_summed,
            "L2_summed": self.L2_loss_summed,
            "NRMSE": self.NRMSE,
            "RMSE": self.RMSE,
            "L1_masked": self.L1_masked,
            "L2_masked": self.L2_masked
        }

    def generalize_dice_loss(self, one_hot, logits):
        w = tf.reduce_sum(one_hot, axis=[1, 2, 3])
        w = 1 / (w ** 2 + 1e-6)
        # w = w / tf.reduce_sum(w)  # Normalize weights

        probs = tf.nn.softmax(logits)

        multed = tf.reduce_sum(probs * one_hot, axis=[1, 2, 3])
        summed = tf.reduce_sum(probs, axis=[1, 2, 3]) + tf.reduce_sum(one_hot, axis=[1, 2, 3])

        numerator = w * multed
        denominator = w * summed

        dice_score = tf.reduce_mean(2. * numerator / (denominator + 1e-6), axis=0)

        return 1. - tf.reduce_mean(dice_score)

    def dice_score_from_logits(self, one_hot, logits, probs=False):
        """
        Dice coefficient (F1 score) is between 0 and 1.
        :param labels: one hot encoding of target (num_samples, num_classes)
        :param logits: output of network (num_samples, num_classes)
        :return: Dice score by each class
        """

        probs = tf.nn.softmax(logits) if not probs else logits

        # Axes which don't contain batches or classes (i.e. exclude first and last axes)
        target_axes = list(range(len(probs.shape)))[1:-1]

        intersect = tf.reduce_sum(probs * one_hot, axis=target_axes)
        denominator = tf.reduce_sum(probs, axis=target_axes) + tf.reduce_sum(one_hot, axis=target_axes)

        dice_score = tf.reduce_mean(2. * intersect / (denominator + 1e-6), axis=0)

        return dice_score

    def dice_loss(self, one_hot, logits, probs=False):
        return 1. - tf.reduce_mean(self.dice_score_from_logits(one_hot, logits, probs=probs))

    def L1_loss(self, labels, logits):
        return tf.reduce_mean(tf.math.abs(labels - logits))

    def L1_loss_summed(self, labels, logits, axis=(1, 2, 3)):
        return tf.reduce_mean(tf.reduce_sum(tf.math.abs(labels - logits), axis=axis))

    def L2_loss(self, labels, logits):
        return tf.reduce_mean(tf.math.square(labels - logits))

    def L2_loss_summed(self, labels, logits, axis=(1, 2, 3)):
        return tf.reduce_mean(tf.reduce_sum(tf.math.square(labels - logits), axis=axis))

    def L1_masked(self, labels, logits, mask):
        return tf.math.reduce_sum(tf.math.abs(labels*mask - logits*mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)

    def L2_masked(self, labels, logits, mask):
        return tf.math.reduce_sum(tf.math.square(labels*mask - logits*mask)) / tf.cast(tf.math.count_nonzero(mask), tf.float32)

    def NRMSE(self, labels, logits, mask):
        labels = labels * mask if mask is not None else labels
        logits = logits * mask if mask is not None else logits

        mask = tf.ones_like(labels) if mask is None else mask

        true_flat = tf.keras.layers.Flatten()(labels)
        fake_flat = tf.keras.layers.Flatten()(logits)
        mask_flat = tf.keras.layers.Flatten()(mask)

        # Get only elements in mask
        true_new = tf.boolean_mask(true_flat, mask_flat)
        fake_new = tf.boolean_mask(fake_flat, mask_flat)

        # Demean
        true_demean = true_new - tf.math.reduce_mean(true_new)
        fake_demean = fake_new - tf.math.reduce_mean(fake_new)

        return 100 * tf.norm(true_demean - fake_demean) / tf.norm(true_demean)

    def RMSE(self, labels, logits, mask):
        return 100 * tf.norm(labels - logits) / tf.norm(labels)
