import os
from cca_score import CCA
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class GetActivations(nn.Module):
    """
    Class for getting activations from a model.
    """

    def __init__(self, model):
        super(GetActivations, self).__init__()
        self.model = model

    def forward(self, x):
        out = x.permute(0, 2, 1)
        activations = []
        model_front = self.model.model.front

        x = out.unsqueeze(dim=1)

        out = model_front.relu(model_front.bn1(model_front.conv1(x)))
        activations.append({"first relu": out})

        for name, layer in model_front.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                for sec_name, sec_layer in layer.named_children():
                    identity = out

                    out = sec_layer.relu(sec_layer.bn1(sec_layer.conv1(out)))
                    activations.append({f"{name} relu": out})

                    out = sec_layer.bn2(sec_layer.conv2(out))
                    out = sec_layer.SimAM(out)
                    activations.append({"SimAM": out})

                    if sec_layer.downsample is not None:
                        identity = sec_layer.downsample(identity)

                    out += identity
                    out = sec_layer.relu(out)
                    activations.append({f"{name} relu": out})

        out = self.model.model.pooling(out)
        activations.append({"pooling": out})

        if self.model.model.drop:
            out = self.model.model.drop(out)

        out = self.model.model.bottleneck(out)

        return activations, out


def get_cca(acts, encoder):
    """
    Computes CCA similarity between activation and one-hot label vector.
    """
    cca_coefs = []
    oh_label = acts['label']
    label = encoder.inverse_transform(oh_label).item()
    layers = [list(item.keys())[0] for item in acts['act']]

    for act in acts['act']:
        for act_value in act.values():
            act_new = act_value.cpu().view(64, -1).numpy()

            labels_repeated = np.tile(
                oh_label.flatten(), (2, act_new.shape[1])
            )
            labels_repeated = labels_repeated[:, :act_new.shape[1]]

            cca = CCA(act_new, labels_repeated)

            cca_results = cca.get_cca_parameters(
                epsilon_x=1e-4,
                epsilon_y=1e-4,
                verbose=False
            )

            cca_coefs.append(np.mean(cca_results["cca_coef1"]))
    return {'label': label, 'cca': cca_coefs}, layers


def visualize_cca_score(cca_coefs, save_path):
    """
    Сохраняет визуализацию одного усреднённого значения CCA-коэффициентов по всем классам.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Собираем все коэффициенты в массив
    all_coefs = list(cca_coefs.values())  # список массивов

    # Приводим к единому numpy-массиву (размер: классы x слои)
    all_coefs = np.array(all_coefs)

    # Усредняем по классам
    mean_coef = np.mean(all_coefs, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(mean_coef) + 1), mean_coef, label=f"Mean CCA (avg: {np.mean(mean_coef):.2f})")

    plt.title("Average CCA Similarity Across All Classes")
    plt.xlabel("Layer number")
    plt.ylabel("CCA similarity")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def save_cca(cca, layers, save_path):
    """
    Saves computed CCA similarity in .txt file for each layer.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for label, coef in cca.items():
            f.write(f"{label}\n")
            for i in range(len(coef)):
                f.write(f"{layers[i]}: {coef[i]}\n")
