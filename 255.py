import numpy as np 
from sklearn import preprocessing

raw_data = np.loadtxt('352 - Audiobooks-data.csv', delimiter=',')
unscaled_inputs_all = raw_data[:, 1:-1]
targets_all = raw_data[:,-1]

#Shuffle 

shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)
unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
targets_all = targets_all[shuffled_indices]

#Balance The Dataset

num_one_count = int(np.sum(targets_all))
zero_targets_count = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_count += 1
        if zero_targets_count > num_one_count:
            indices_to_remove.append(i)


unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

# Standardize the inputs

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# Shuffle the Data

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#split the data

sample_count = shuffled_inputs.shape[0]
train_sample_count = int(0.8 *sample_count)
validation_sample_count = int(0.1*sample_count)
test_sample_count = sample_count - train_sample_count - validation_sample_count

train_inputs = shuffled_inputs[:train_sample_count]
train_targets = shuffled_targets[:train_sample_count]

validation_inputs = shuffled_inputs[train_sample_count:train_sample_count+validation_sample_count]
validation_targets = shuffled_targets[train_sample_count:train_sample_count+validation_sample_count]

test_inputs = shuffled_inputs[train_sample_count+validation_sample_count:]
test_targets = shuffled_targets[train_sample_count+validation_sample_count:]

print(np.sum(train_targets), train_sample_count, np.sum(train_targets) / train_sample_count)
print(np.sum(validation_targets), validation_sample_count, np.sum(validation_targets) / validation_sample_count)
print(np.sum(test_targets), test_sample_count, np.sum(test_targets) / test_sample_count)

# save datasets inn *.npz

np.savez('Audiobooks_data_train', inputs= train_inputs, targets= train_targets)
np.savez('Audiobooks_data_validation', inputs= validation_inputs, targets= validation_targets)
np.savez('Audiobooks_data_test', inputs= test_inputs, targets= test_targets)