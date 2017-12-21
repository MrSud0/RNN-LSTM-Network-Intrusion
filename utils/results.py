

#Module that returns the results of the statistical results
import argparse

from utils.preprocess  import plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Confusion Matrix for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-t', '--training_results_path', required=False, type=str,
                       help='path where the results of model training are stored')
    group.add_argument('-v', '--test_results_path', required=True, type=str,
                       help='path where the results of the model testing are stored')
    arguments = parser.parse_args()
    return arguments


def main(argv):
    if(argv.training_results_path!=None):
        training_confusion_matrix = plot_confusion_matrix(phase='Training', path=argv.training_results_path,
                                                      class_names=['normal', 'anomaly'])

    testing_confusion_matrix = plot_confusion_matrix(phase='Testing', path=argv.test_results_path,
                                                        class_names=['normal', 'anomaly'])
    # display the findings from the confusion matrix
    print('True negative : {}'.format(training_confusion_matrix[0][0][0]))
    print('False negative : {}'.format(training_confusion_matrix[0][1][0]))
    print('True positive : {}'.format(training_confusion_matrix[0][1][1]))
    print('False positive : {}'.format(training_confusion_matrix[0][0][1]))
    print('Training accuracy : {}'.format(training_confusion_matrix[1]))

    # display the findings from the confusion matrix
    print('True negative : {}'.format(testing_confusion_matrix[0][0][0]))
    print('False negative : {}'.format(testing_confusion_matrix[0][1][0]))
    print('True positive : {}'.format(testing_confusion_matrix[0][1][1]))
    print('False positive : {}'.format(testing_confusion_matrix[0][0][1]))
    print('Testing accuracy : {}'.format(testing_confusion_matrix[1]))


if __name__ == '__main__':
    args = parse_args()

    main(args)
