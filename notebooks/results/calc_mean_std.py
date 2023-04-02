import glob

import pandas as pd


def calculate_accuracy():
    precision = float(input('Precision: '))
    recall = float(input('Recall: '))

    accuracy = 0.5 - (recall / (2 * precision)) + recall

    print(f'Accuracy: {accuracy}')

    return accuracy


if __name__ == '__main__':
    path = r'./*.csv'
    files = glob.glob(path)

    for file in files:

        print(file)

        name = file[2:-4]

        df = pd.read_csv(name + '.csv', usecols=['dataset', 'model', 'accuracy', 'precision', 'recall'])

        grouped_mean = df.groupby(['dataset', 'model']).agg(['mean', 'std']).reset_index()


        print(grouped_mean)

        grouped_mean.to_csv('mean_and_standard_deviation/' + name + '_mean.csv', index=False, na_rep='NaN', float_format='%.3f')

