import torch
# from base_Resnet import ResNet50Custom
# from dataset_CelebA import trainloader, validloader, testloader
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
import csv
from networks.Resnet_alligned2 import *

from data_handler.dataloader_factory import DataloaderFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tmp = DataloaderFactory.get_dataloader('celeba', img_size=176,
                                                        batch_size=100, seed=0,
                                                        num_workers=4,
                                                        target='Wearing_Necklace',
                                                        skew_ratio=1,
                                                        labelwise=True
                                                        )

num_classes, num_groups, train_loader, testloader = tmp




# def test(model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     activations_dict = {}
#
#     with torch.no_grad():
#         for batch_idx, (data, _, attr, target, (sensitive, img_name)) in enumerate(test_loader):
#             data, target, attr = data.to(device), target.to(device), attr.to(device)
#             outputs = model(data,get_inter = True )
#             test_loss += criterion(outputs[-1], target).item()  # sum up batch loss
#             pred = outputs[-1].argmax(dim=1)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             # Store results in the dictionary
#             # Note: You might need to detach and convert tensors to the appropriate format
#             for i in range(data.size(0)):  # Iterate through each sample in the batch
#                 sample_id = batch_idx * test_loader.batch_size + i
#                 activations_dict[sample_id] = {
#                     'activations': outputs[0][i].cpu().numpy(), #change for different layers
#                     'image_ID' : img_name[i],
#                     'label': target[i].item(),
#                     'attribute': attr[i].item()
#                 }
#
#         a = len(test_loader.dataset)
#         test_loss /= len(test_loader.dataset)
#         accuracy = 100. * correct / len(test_loader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset), accuracy))
#
#     return accuracy, activations_dict



def check_accuracy(model, device, loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    activations_dict = {}

    with torch.no_grad():  # No need to track gradients for validation
        for batch_idx, (data, _, attr, target, (index, image_name)) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data, get_inter=True)
            _, predicted = torch.max(outputs[-1].data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            for i in range(data.size(0)):  # Iterate through each sample in the batch
                sample_id = batch_idx * loader.batch_size + i
                activations_dict[image_name[i]] = {
                    'activations': outputs[3][i].cpu().numpy(),  # change for different layers
                    'image_ID': image_name[i],
                    'label': target[i].item(),
                    'attribute': attr[i].item()
                }

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')
    return accuracy, activations_dict

criterion = nn.CrossEntropyLoss()

model = resnet18(pretrained=False, num_classes = 2)

outputs = model(torch.rand([1,3,176,176]), get_inter = True)
layer_list = []
teacher_feature_size = outputs[0].size(1)
student_feature_size = outputs[0].size(1)
layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
model.adaptation_layers = nn.ModuleList(layer_list)
model.adaptation_layers.cuda()

model.load_state_dict(torch.load('/ubc/ece/home/ra/grads/nourhanb/Documents/distil/Fairify/trained_models/170624_Wearing_Necklace/celeba/scratch/best_model_acc_wearing_necklace_0.57.pth'))
model.to(device)
model.eval()



accuracy, dict = check_accuracy(model, device, train_loader )


activations_list = [v['activations'].flatten() for v in dict.values()]
stacked_activations = np.stack(activations_list)


# Perform K-Means clustering
k = 8  # Number of clusters,
kmeans = KMeans(n_clusters=k,  random_state=42)
kmeans.fit(stacked_activations)

# Get cluster labels for each data point
labels = kmeans.labels_


for i, key in enumerate(dict.keys()):
    dict[key]['cluster_label'] = labels[i]


# count = 0
# count2 = 0
# for entry in dict.values():
#     if entry['attribute'] == entry['cluster_label']:
#         count+=1
#     if entry['label'] == entry['cluster_label']:
#         count2+=1


for sub_dict in dict.values():
    if sub_dict:
        first_key = next(iter(sub_dict))
        del sub_dict[first_key]

fieldnames = list(next(iter(dict.values())).keys())

# Writing to csv file
with open('Wearing_Necklace_layer4_k8.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Writing headers
    writer.writeheader()

    # Writing data rows
    for key in dict:
        writer.writerow(dict[key])
