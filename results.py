import numpy as np
import scipy.io

savepath = '/ubc/ece/home/ra/grads/nourhanb/Documents/distil/Fairify/results/260624_Wearing_Necklace/celeba/kd_mfd/best_student_model_acc_wearing_necklace_no_kl_0.71._test_pred'
data = scipy.io.loadmat(savepath)


def softmax(logits):
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=-1, keepdims=True)

probabilities = softmax(data['output_set'])
predictions = np.argmax(probabilities, axis=1)
accuracy = np.mean(predictions == data['target_set']) * 100

# Example data (replace these with your actual data arrays)

group_set = data['group_set']   # Group labels (0 or 1)
target_set = data['target_set']  # Actual labels


predictions = np.squeeze(predictions)
group_set = np.squeeze(group_set)
target_set = np.squeeze(target_set)

# Function to calculate accuracy
def calculate_accuracy(preds, targets):
    if len(preds) == 0:  # To handle any case where there might be no data for a combination
        return None
    return np.mean(preds == targets) * 100

# Calculating accuracy for each group-target combination
results = {}
for group in [0, 1]:
    for target in [0, 1]:
        mask = (group_set == group) & (target_set == target)
        preds_filtered = predictions[mask]  # Ensure predictions is correctly subsetted
        target_filtered = target_set[mask]  # Same for target_set
        acc = calculate_accuracy(preds_filtered, target_filtered)
        results[f'Group {group} and Target {target}'] = f'{acc:.2f}%' if acc is not None else 'No Data'

# Printing results
for key, value in results.items():
    print(f"Accuracy for {key}: {value}")


print("Over all accuracy is ", accuracy)
