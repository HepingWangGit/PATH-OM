import numpy as np
from sklearn.metrics import accuracy_score

def classification_metric(predicted, true, bins):

    # Validate that predicted and true arrays have the same length
    if predicted.shape[0] != true.shape[0]:
        raise ValueError("Predicted and true arrays must have the same length.")

    # Assign classes based on bins using np.digitize
    predicted_classes = np.digitize(predicted, bins) - 1  # digitize starts at 1
    true_classes = np.digitize(true, bins) - 1

    num_bins = len(bins) - 1

    # Handle edge cases where values might be exactly equal to the last bin edge
    predicted_classes[predicted_classes == num_bins] = num_bins - 1
    true_classes[true_classes == num_bins] = num_bins - 1

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    #print(f"Classification Accuracy: {accuracy * 100:.2f}%")

    return accuracy

def count_consecutive_elements(input_list):
        if not input_list:
            return []

        output = []
        current_element = input_list[0]
        count = 1

        for element in input_list[1:]:
            if element == current_element:
                count += 1
            else:
                output.append((count, current_element))
                current_element = element
                count = 1

        # Append the last element and its count
        output.append((count, current_element))

        return output

def multivariate_r2(y_true, y_pred):
    # Calculate residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares
    mean = np.mean(y_true)
    ss_tot = np.sum((y_true - mean) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_res / ss_tot)

    return r2