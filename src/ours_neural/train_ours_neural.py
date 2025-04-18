import torch.optim as optim

from loss.loss import BCELossWithClassWeights
from metrics.helper import print_metrics
from metrics.metrics_calculator import MetricsCalculator
from wiring import get_source_data, get_training_data, get_model

import numpy as np
from data.binvox_rw import write, Voxels
import torch

def train_ours_neural(object_name, query, dimension, metrics_registry):
    print(f"oursNeural {object_name} {dimension}D {query} query")

    # hyperparameters
    n_regions = 50_000
    n_samples = 1500 if dimension == 4 else 500

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)

    # initialise model
    model = get_model(query=query, dimension=dimension)

    # initialise asymmetric binary cross-entropy loss function, and optimiser
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # initialise counter and print_frequency
    weight_schedule_frequency = 250_000
    # total_iterations = weight_schedule_frequency * 200  # set high iterations for early stopping to terminate training
    total_iterations = 10000
    evaluation_frequency = weight_schedule_frequency // 5
    print_frequency = 1000  # print loss every 1k iterations

    # instantiate count for early stopping
    count = 0

    for iteration in range(total_iterations):
        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
                                              n_samples=n_samples)

        # forward pass
        output = model(features)
        # print("feat")
        # print(features.shape)
        # print("target")
        # print(targets.shape)
        # print(output)

        # compute loss
        loss = criterion(output, targets)

        # zero gradients, backward pass, optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # print loss
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f'Iteration: {iteration + 1}, Loss: {loss.item()}')

        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
            print_metrics(metrics)

        if (iteration + 1) % evaluation_frequency == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

            # if convergence to FN 0 is not stable yet and still oscillating
            # let the model continue training
            # by resetting the count
            if count != 0 and metrics["false negatives"] != 0.:
                count = 0

            # ensure that convergence to FN 0 is stable at a sufficiently large class weight
            if metrics["false negatives"] == 0.:
                count += 1

            if count == 3:
                # save final training results
                metrics_registry.metrics_registry["oursNeural"] = {
                    "class weight": class_weight,
                    "iteration": iteration+1,
                    "false negatives": metrics["false negatives"],
                    "false positives": metrics["false positives"],
                    "true values": metrics["true values"],
                    "total samples": metrics["total samples"],
                    "loss": f"{loss:.5f}"
                }

                # early stopping
                print("early stopping\n")
                break

        # schedule increases class weight by 20 every 500k iterations
        if (iteration + 1) % weight_schedule_frequency == 0 or iteration == 0:
            if iteration == 0:
                pass
            elif (iteration + 1) == weight_schedule_frequency:
                class_weight = 20
            else:
                class_weight += 20

            criterion.negative_class_weight = 1.0 / class_weight

            print("class weight", class_weight)
            print("BCE loss negative class weight", criterion.negative_class_weight)
    # print(data.shape)
    final_features = []
    points = []
    for x in range(32):
        for y in range(32):
            for z in range(32):
                points.append([x, y, z]) # points in the binvox space as a grid of 32x32x32 points
                final_features.append([x/32.0, y/32.0, z/32.0]) # points scaled down to the [0, 1) space expected by the model
    
    final_pred = (model(torch.tensor(final_features).cuda()).cpu().detach() >= 0.5).float().numpy() # 0 or 1 predictions on each grid point
    final = np.zeros((32, 32, 32), dtype=np.bool) # to store final voxels for binvox file
    for point, prediction in zip(points, final_pred):
        final[point[0]][point[1]][point[2]] = bool(prediction)
    filepath = "../../testNeuralBounding.binvox"
    with open(filepath, 'w', encoding="latin-1") as fp:
        write(Voxels(final, [32, 32, 32], [0.0, 0.0, 0.0], 1.0, 'xyz'), fp) # write binvox file
        
# import torch.optim as optim
# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname('src'), '..')))

# from loss.loss import BCELossWithClassWeights
# from metrics.helper import print_metrics
# from metrics.metrics_calculator import MetricsCalculator
# from wiring import get_source_data, get_training_data, get_model


# def train_ours_neural(object_name, query, dimension, metrics_registry):
#     print(f"oursNeural {object_name} {dimension}D {query} query")

#     # hyperparameters
#     n_regions = 32768 #50_000
#     n_samples = 1500 if dimension == 4 else 500

#     # load data
#     data = get_source_data(object_name=object_name, dimension=dimension) # 32 by 32 by 32

#     # Plot the input data
#     # data2 =data.cpu().detach().numpy()
#     # values = data2.flatten()
#     # x, y, z = np.indices(data2.shape)
#     # x_flat = x.flatten()
#     # y_flat = y.flatten()
#     # z_flat = z.flatten()
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.scatter(x_flat, y_flat, z_flat, c=values, cmap='gray', marker='o') 
#     # plt.show()

#     # initialise model
#     model = get_model(query=query, dimension=dimension)

#     # initialise asymmetric binary cross-entropy loss function, and optimiser
#     class_weight = 1
#     criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
#     optimiser = optim.Adam(model.parameters(), lr=0.0001)

#     # initialise counter and print_frequency
#     weight_schedule_frequency = 250_000
#     total_iterations = 100000 #weight_schedule_frequency * 200  # set high iterations for early stopping to terminate training
#     evaluation_frequency = weight_schedule_frequency // 5
#     print_frequency = 1000  # print loss every 1k iterations

#     # instantiate count for early stopping
#     count = 0

#     for iteration in range(total_iterations):
#         features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
#                                               n_samples=n_samples)

#         # forward pass
#         output = model(features)
#         # print(output.shape) # 50,000 by 1 -> 32 by 32 by 32

#         # compute loss
#         loss = criterion(output, targets)

#         # zero gradients, backward pass, optimiser step
#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         # print loss
#         if (iteration + 1) % print_frequency == 0 or iteration == 0:
#             print(f'Iteration: {iteration + 1}, Loss: {loss.item()}')

#         #if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
#         if iteration == total_iterations-1:
#             prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
#             # Plot values in predictions when n_regions = input data dimension 
#             predictionsReshaped = prediction.reshape(32, 32, 32)
#             values = predictionsReshaped.flatten()
#             x, y, z = np.indices(predictionsReshaped.shape)
#             x_flat = x.flatten()
#             y_flat = y.flatten()
#             z_flat = z.flatten()
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(x_flat, y_flat, z_flat, c=values, cmap='gray', marker='o') 
#             plt.show()
#             target = targets.cpu().detach().numpy()
#             metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
#             print_metrics(metrics)

#         if (iteration + 1) % evaluation_frequency == 0:
#             prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
#             target = targets.cpu().detach().numpy()
#             metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

#             # if convergence to FN 0 is not stable yet and still oscillating
#             # let the model continue training
#             # by resetting the count
#             if count != 0 and metrics["false negatives"] != 0.:
#                 count = 0

#             # ensure that convergence to FN 0 is stable at a sufficiently large class weight
#             if metrics["false negatives"] == 0.:
#                 count += 1

#             if count == 3:
#                 # save final training results
#                 metrics_registry.metrics_registry["oursNeural"] = {
#                     "class weight": class_weight,
#                     "iteration": iteration+1,
#                     "false negatives": metrics["false negatives"],
#                     "false positives": metrics["false positives"],
#                     "true values": metrics["true values"],
#                     "total samples": metrics["total samples"],
#                     "loss": f"{loss:.5f}"
#                 }

#                 # early stopping
#                 print("early stopping\n")
#                 break

#         # schedule increases class weight by 20 every 500k iterations
#         if (iteration + 1) % weight_schedule_frequency == 0 or iteration == 0:
#             if iteration == 0:
#                 pass
#             elif (iteration + 1) == weight_schedule_frequency:
#                 class_weight = 20
#             else:
#                 class_weight += 20

#             criterion.negative_class_weight = 1.0 / class_weight

#             print("class weight", class_weight)
#             print("BCE loss negative class weight", criterion.negative_class_weight)

