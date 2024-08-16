import random
import torch


def k_fold_cv_iter(k: int, training_features: torch.Tensor, training_labels: torch.Tensor):
    if k < 1:
        k = 5
    length = len(training_features)
    num_every_part = length // k

    indices = [i for i in range(length)]
    random.shuffle(indices)
    training_features = training_features[indices]
    training_labels = training_labels[indices]

    for epoch in range(k):
        start_val = epoch * num_every_part
        end_val = (epoch + 1) * num_every_part if epoch != k - 1 else length

        final_vali_features = training_features[start_val:end_val]
        final_vali_labels = training_labels[start_val:end_val]

        final_training_features = torch.cat((training_features[:start_val], training_features[end_val:]), dim=0)
        final_training_labels = torch.cat((training_labels[:start_val], training_labels[end_val:]), dim=0)

        yield final_training_features, final_training_labels, final_vali_features, final_vali_labels


