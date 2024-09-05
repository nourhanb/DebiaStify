import torch
import networks
import data_handler
import torch.nn as nn
from utils import check_log_dir, make_log_name, set_seed


seed = 0
set_seed(seed)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelpath = 'trained_models/280424-4/celeba/kd_mfd/resnet_seed0_epochs50_bs128_lr0.001_rbf_sigma1.0_labelwise_temp3_lambh0.0_lambf7.0_fixedlamb.pt'
model = networks.ModelFactory.get_model('resnet', 2, 176, pretrained=False)
model.load_state_dict(torch.load(modelpath))
model.to(device)
dataset = 'celeba'

tmp = data_handler.DataloaderFactory.get_dataloader(dataset, img_size=176,
                                                        batch_size=128, seed=0,
                                                        num_workers=2,
                                                        target='Attractive',
                                                        skew_ratio=1,
                                                        labelwise=True
                                                        )
num_classes, num_groups, train_loader, test_loader = tmp



def test_combination(label_val, attribute_val):
    correct = 0
    total = 0
    # correct = [0 for _ in range(2)]
    # predicted = [0 for _ in range(2)]
    model.eval()
    with torch.no_grad():
        for images, _, attributes, labels, (index, img_name) in test_loader:  # Assuming test_loader is already filtered or you filter here
            # Filter data based on current combination
            mask = (labels == label_val) & (attributes == attribute_val)
            mask = mask.to(device)
            if torch.any(mask):

                images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)
                images, labels, attributes = images[mask], labels[mask], attributes[mask]

                # Forward pass

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                # for classifier_index in range(len(outputs)):
                    # _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    # correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Testing combination Label={label_val}, Attribute={attribute_val}: Accuracy = {accuracy:.2f}%')


for label in [0, 1]:
    for attribute in [0, 1]:
        test_combination(label, attribute)


# def test_accuracy():
#     correct = 0
#     total = 0
#     # correct = [0 for _ in range(2)]
#     # predicted = [0 for _ in range(2)]
#     with torch.no_grad():
#         for images, _, attributes, labels, (index, img_name) in test_loader:  # Assuming test_loader is already filtered or you filter here
#             # Filter data based on current combination
#
#             images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)
#
#
#             # Forward pass
#
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total if total > 0 else 0
#     print(f'Accuracy = {accuracy:.2f}%')
#
#
#
#
# test_accuracy()

def evaluate(model, loader, criterion, device=None, groupwise=False):
    model.eval()
    num_groups = loader.dataset.num_groups
    num_classes = loader.dataset.num_classes
    device = device if device is None else device

    eval_acc = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
    eval_loss = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
    eval_eopp_list = torch.zeros(num_groups, num_classes).cuda(device)
    eval_data_count = torch.zeros(num_groups, num_classes).cuda(device)

    if 'Custom' in type(loader).__name__:
        loader = loader.generate()
    with torch.no_grad():
        for j, eval_data in enumerate(loader):
            # Get the inputs
            inputs, _, groups, classes, _ = eval_data
            #
            labels = classes

            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            groups = groups.cuda(device)

            outputs = model(inputs)

            if groupwise:
                groups = groups.cuda(device)
                loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_loss[g, l] += loss[(groups == g) * (labels == l)].sum()
                        eval_acc[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

            else:
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                eval_acc += acc.sum()

                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

        eval_loss = eval_loss / eval_data_count.sum() if not groupwise else eval_loss / eval_data_count
        eval_acc = eval_acc / eval_data_count.sum() if not groupwise else eval_acc / eval_data_count
        eval_eopp_list = eval_eopp_list / eval_data_count
        eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
        eval_max_eopp = torch.max(eval_max_eopp).item()
    model.train()
    return eval_loss, eval_acc, eval_max_eopp

criterion = torch.nn.CrossEntropyLoss()


eval_loss, eval_acc, eval_deopp = evaluate(model, test_loader, criterion, device= device)

print(1)
