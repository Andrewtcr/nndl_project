from diffuser import diffuse_generator as dgen
import csv
import shutil


Y_TRAIN_PATH = 'neuralnet_subclasses_competition_data/y_train.csv'
Y_GENERATED_PATH = 'neuralnet_subclasses_competition_data/y_generated.csv'
X_GENERATED_PATH = 'neuralnet_subclasses_competition_data/x_generated'

AMT_TO_GEN = 50
TARGET_DIMS = (8, 8)


subclasses = None
with open(Y_TRAIN_PATH, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Labels are in the header
    subclasses = next(csvreader)

# print(len(subclasses))

# with open(Y_GENERATED_PATH, newline='', mode='w') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')

#     csvwriter.writerow(subclasses)

#     for i in range(len(subclasses)):
#         for _ in range(50):
#             row = [0] * len(subclasses)
#             row[i] = 1

#             csvwriter.writerow(row)


exit() # Just in case

with open(Y_GENERATED_PATH, newline='', mode='w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')

    csvwriter.writerow(subclasses)

    generator = dgen()

    j = 0
    for label in subclasses:
        print(label)

        for i in range(AMT_TO_GEN):

            output = generator.pipe(label, num_inference_steps=25).images[0]
            output = output.resize(TARGET_DIMS)

            # Save to directory
            # Will be zipped later
            output.save(f'{X_GENERATED_PATH}/{j + i}.png')

            # Also, remember to make labels
            labels = [0] * len(subclasses)
            labels[i] = 1
            csvwriter.writerow(labels)
        
        j += AMT_TO_GEN

shutil.make_archive(X_GENERATED_PATH, 'zip', 'generated_shuffle')