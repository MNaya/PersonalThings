import pickle
import numpy as np
import random as rd
from matplotlib import pyplot as plt


# ---------------------------  Class for storing the data obtained during the full evolution  --------------------------
class classSaveData(object):
    def __init__(self, numGen):

        self.Flag = False
        self.species = []
        self.contGen = 0

        self.evhist = []
        self.best_gs = []
        self.best_fit_ever = 0
        self.dev_stage = 0
        self.generation = 0
        self.modelHandle = 0
        self.max_generations = 0
        self.perc75 = [0 for n in xrange(0, numGen)]
        self.perc25 = [0 for n in xrange(0, numGen)]
        self.valueDistance = [0 for n in xrange(0, numGen)]
        self.valueFitnessMax = [0 for n in xrange(0, numGen)]
        self.fitnessValidation = [0 for n in xrange(0, numGen)]
        self.valueFitnessMedia = [0 for n in xrange(0, numGen)]
        self.valueFitnessMediana = [0 for n in xrange(0, numGen)]
        self.bestIndividualsInGeneration = [0 for n in xrange(0, numGen)]
# ----------------------------------------------------------------------------------------------------------------------


# Lineal normalization between the extreme values of the variable, ang the extreme data of the sigmoid (in this case: -3,3:
def normalizeInputs (clase, dato_in):
    sig_sup = 3.0
    sig_inf = -3.0

    coord_inf = [0]
    coord_sup = [0]

    coord_inf[0] = clase.y_min
    coord_sup[0] = clase.y_max

    aux = [((((dato_in[n] - coord_inf[n]) * (sig_sup - sig_inf)) / (coord_sup[n] - coord_inf[n])) + sig_inf) for n in xrange(0, len(dato_in))]

    return aux
# ----------------------------------------------------------------------------------------------------------------------


# --------------  Normalize the output variable for been used for training in Keras  -----------------
def normalizeTargets (clase, dato_in):
    sig_sup = 1.0
    sig_inf = 0.0

    ang_inf = [0]
    ang_sup = [0]

    ang_sup[0] = clase.x_max
    ang_inf[0] = clase.x_min


    aux = [((((dato_in[n] - ang_inf[n]) * (sig_sup - sig_inf)) / (ang_sup[n] - ang_inf[n])) + sig_inf) for n in xrange(0, len(dato_in))]

    return aux
# ----------------------------------------------------------------------------------------------------------------------


# --------------------------------  Denormaliza the data obteined on the net ----------------------------------------------------
def deNormalizeOuputs(clase, dato_out):
    sig_sup = 1.0
    sig_inf = 0.0

    ang_inf = [0]
    ang_sup = [0]

    ang_sup[0] = clase.x_max
    ang_inf[0] = clase.x_min

    aux = [((((dato_out[n] - sig_inf) * (ang_sup[n] - ang_inf[n])) / (sig_sup - sig_inf)) + ang_inf[n]) for n in xrange(0, len(dato_out))]

    return aux
# ----------------------------------------------------------------------------------------------------------------------


# -------  Function for creating the data that would be used for training the data, or for use it in evolution  --------
def createAllSetsOfData (clase, aa, bb, cc, dd, ee):

    x_array = np.arange(clase.x_min, clase.x_max, 0.01)     # Creation of the array of "X" values

    Y_norm = []
    X_norm = []
    raw_Ycoord = []
    raw_Xangles = []
    for x in x_array:
        angles = [x]
        aux_y = [0]

        aux_y[0] = aa + bb*x + cc*x*x + dd*x*x*x + ee*x*x*x*x

        Y_norm.append(normalizeInputs(clase, aux_y))    # Array of "Y" values normalized
        X_norm.append(normalizeTargets(clase, angles))  # Array of "X" normalized for being used in the training with Keras
        raw_Ycoord.append(aux_y)                        # Array of "Y" values without been normalized
        raw_Xangles.append(angles)                      # Array of "X" values without been normalized


    # Selection of those positions for:
    # - Training.
    # - Validation.
    # - Test.
    sampleSize = len(Y_norm)
    train_number = int(len(Y_norm) * 0.7)

    # Positions for trainning
    Train_positions = rd.sample(range(sampleSize), int(train_number))
    Train_positions.sort()
    remainingPositions = []

    counter = 0
    data = 0

    while counter < (len(Y_norm)):
        if (counter - data) > len(Train_positions)-1:
            remainingPositions.append(counter)
            counter += 1
        else:
            if counter == Train_positions[counter - data]:
                counter += 1
            else:
                remainingPositions.append(counter)
                data += 1
                counter += 1

    print len(remainingPositions)
    print int(len(remainingPositions)*0.5 -1.0)

    # Positions for validation
    Validation_positions = rd.sample(remainingPositions, int(len(remainingPositions)*0.5))
    Validation_positions.sort()
    Test_positions = []
    counter = 0
    data = 0

    # Positions for testing
    while counter < (len(remainingPositions)-1):
        if (counter - data) > len(Validation_positions)-1:
            Test_positions.append(remainingPositions[counter])
            counter += 1
        else:
            if remainingPositions[counter] == Validation_positions[counter - data]:
                counter += 1
            else:
                Test_positions.append(remainingPositions[counter])
                counter += 1
                data += 1


    # Data generation for trainning
    counter = 0
    norm_X_train = []
    coord_Y_train = []
    Initial_X_train = []
    norm_coord_Y_train = []
    for train in Train_positions:
        coord_Y_train.append(raw_Ycoord[train])
        Initial_X_train.append(raw_Xangles[train])
        norm_coord_Y_train.append(Y_norm[train])
        norm_X_train.append(X_norm[train])
        counter += 1

    inputs_Keras_train = np.asarray(norm_coord_Y_train)
    output_Keras_train = np.asarray(norm_X_train)

    # Data generation for validation
    counter = 0
    norm_X_val = []
    coord_Y_val = []
    Initial_X_val = []
    norm_coord_Y_val = []
    for validation in Validation_positions:
        coord_Y_val.append(raw_Ycoord[validation])
        Initial_X_val.append(raw_Xangles[validation])
        norm_coord_Y_val.append(Y_norm[validation])
        norm_X_val.append(X_norm[validation])
        counter += 1

    inputs_Keras_validation = np.asarray(norm_coord_Y_val)
    output_Keras_validation = np.asarray(norm_X_val)

    # Data generation for testing
    counter = 0
    norm_X_test = []
    coord_Y_test = []
    Initial_X_test = []
    norm_coord_Y_test = []
    for test in Test_positions:
        coord_Y_test.append(raw_Ycoord[test])
        Initial_X_test.append(raw_Xangles[test])
        norm_coord_Y_test.append(Y_norm[test])
        norm_X_test.append(X_norm[test])
        counter += 1

    inputs_Keras_test = np.asarray(norm_coord_Y_test)
    output_Keras_test = np.asarray(norm_X_test)

    print ('Total size of the sample', sampleSize)
    print ('Train', len(Train_positions))
    print ('Validation', len(Validation_positions))
    print ('Test', len(Test_positions))

    return (coord_Y_train, Initial_X_train, inputs_Keras_train, output_Keras_train), (coord_Y_val, Initial_X_val, inputs_Keras_validation, output_Keras_validation), (coord_Y_test, Initial_X_test, inputs_Keras_test, output_Keras_test)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def saveStatistics (clase, direct):
    text0 = direct + '/' + 'statisticalData' + str(clase.generation - 1) + '.txt'
    auxiliar = [0, 0, 0, 0, 0, 0, 0, 0]
    auxiliar[0] = clase.valueFitnessMax
    auxiliar[1] = clase.valueFitnessMediana
    auxiliar[2] = clase.valueFitnessMedia
    auxiliar[3] = clase.perc75
    auxiliar[4] = clase.perc25
    auxiliar[5] = clase.bestIndividualsInGeneration
    auxiliar[6] = clase.best_gs

    file = open(text0, 'wb')
    pickle.dump(auxiliar, file)
    file.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_history(history, directorio):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.7f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.7f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    aux = directorio + '/' + 'Loss.png'
    plt.savefig(aux)
    plt.close()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    aux = directorio + '/' + 'Accuracy.png'
    plt.savefig(aux)
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def representationTest(dist, texto, prom_media, prom_mediana, limite):
    x = [n for n in xrange(0, len(dist))]
    per75d = [np.percentile(dist, 75) for n in xrange(0, len(dist))]
    per25d = [np.percentile(dist, 25) for n in xrange(0, len(dist))]
    medianad = [np.median(dist) for n in xrange(0, len(dist))]
    mediad = [np.average(dist) for n in xrange(0, len(dist))]
    median_value = mediad[0]

    plt.plot(x, dist, color='blue', linestyle='-', label='Distance mm')
    plt.plot(x, medianad, color='red', linestyle='-', label='Median')
    plt.plot(x, mediad, color='green', linestyle='-', label='Media')
    plt.fill_between(x, per75d, per25d, alpha=0.25, linewidth=0, color='#B22400')
    plt.ylim([0.0, limite/ 10.0])
    text01 = 'Max.Theoric:' + str(limite) + 'Best: ' + str(round(min(dist), 2)) + ' Median: ' + str(round(median_value, 2)) + ' ' + str(prom_mediana) + '% ' + 'Media: ' + str(round(mediad[0], 2)) + ' ' + str(prom_media) + '%'
    plt.title(text01)
    plt.ylabel('Distance value (mm)')
    plt.xlim([0.0, len(dist)])
    plt.xlabel('Proofs')
    plt.legend()
    plt.grid()

    plt.savefig(texto)
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def representationTrain_Validation(clase, direc, limite):
    x = [n for n in xrange(0, clase.max_generations)]
    plt.plot(x, clase.fitnessValidation, color='black', linestyle='-', label='Validation')
    plt.plot(x, clase.valueFitnessMax, color='blue', linestyle='-', label='Maximum')
    plt.plot(x, clase.valueFitnessMediana, color='red', linestyle='-', label='Median')
    plt.plot(x, clase.valueFitnessMedia, color='green', linestyle='-', label='Average')
    plt.fill_between(x, clase.perc75, clase.perc25, alpha=0.25, linewidth=0, color='#B22400')
    plt.ylim([0.0, limite])
    text01 = 'Generation ' + str(clase.generation) + '. ' + ' Best fitness: ' + str(round(clase.valueFitnessMax[clase.generation - 1], 2)) + ' Median: ' + str(round(clase.valueFitnessMediana[clase.generation - 1], 2))
    plt.title(text01)
    plt.ylabel('Fitness values')
    plt.xlim([0.0, clase.max_generations])
    plt.xlabel('Generations')
    plt.legend()
    plt.grid()
    text = direc + '/' + 'feedforward_fitness' + str(clase.generation) + '.png'
    plt.savefig(text)
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------

