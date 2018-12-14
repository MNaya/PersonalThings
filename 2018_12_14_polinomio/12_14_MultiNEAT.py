#  ---------------------------------------------------------------------------------------------------------------------
# Objetivo del programa:
# obtener una red neuronal que funcione de forma similar a la funcion coseno: y = cos(x) para x [-np.pi, np.pi]
# ----------------------------------------------------------------------------------------------------------------------

import os
import cv2
import sys
import errno
import datetime
import copy as cp
import numpy as np
import random as rd
import MultiNEAT as NEAT

import supportFunctions as sf
import multineatConfig as mconfig
from matplotlib import pyplot as plt

from MultiNEAT import EvaluateGenomeList_Serial

# Creation the directory for where saving the data.
day = datetime.date.today().strftime("%d")
year = datetime.date.today().strftime("%Y")
month = datetime.date.today().strftime("%m")
now = datetime.datetime.now()
minutos = now.minute
segundos = now.second

print(datetime.datetime.now())
rand_aux = str(rd.randint(0, 100000))
params = mconfig.parametersCreation()       # Loading the parameters fo the Neat algorithm

directory = month + '_' + day + '_' + str(params.PopulationSize) + 'p_' + '_' + rand_aux

try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Creation of the class for storing the species which are created over evolution
class specie():
    def __init__(self):
        self.tamano = 0
        self.ID = 0


# polynomial and flexible function for testing the functionality of the NEAT


xx_max = 10.0
a = 0.0
b = 1.0
c = 1.0
d = 1.0
e = 1.0

yy_max = a + b * xx_max + c * xx_max * xx_max + d * xx_max * xx_max * xx_max + e * xx_max * xx_max * xx_max * xx_max
topLimit = yy_max

print 'Top limit:', topLimit
print 'Maximo y:', yy_max

# Clase with the maximun and minimun values of "x" and "y"
class variable():
    def __init__(self):
        self.y_min = 0.0
        self.y_max = yy_max
        self.x_min = 0.0
        self.x_max = xx_max

var = variable()

train_set, validation_set, test_set = sf.createAllSetsOfData(var, a, b, c, d, e)# Function where training, validation and

full_generations = 20
datah = sf.classSaveData(full_generations)
datah.max_generations = full_generations

# Variables for testing different NN configurations since the beginning
num_neuron_hidden = 0
num_hidden_layers = 0
seed_gen = 0


# ------------------------------  Function for evaluate a single genome  -----------------------------------------------
def ev_single_genome(genome, test_Y_values):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    net.Flush()
    net_inputs = sf.normalizeInputs(var, test_Y_values)
    net_inputs.append(1.0)

    net.Input(net_inputs)
    net.Activate()
    net_out_std = net.Output()

    X_coord = sf.deNormalizeOuputs(var, net_out_std)

    yy_aux = a + b * X_coord[0] + c * X_coord[0] * X_coord[0] + d * X_coord[0] * X_coord[0] * X_coord[0] + e * X_coord[0] * X_coord[0] * X_coord[0] * X_coord[0]

    polyDiff = abs(yy_aux - test_Y_values[0])
    aux_fitness = topLimit - polyDiff

    return aux_fitness, polyDiff
# ----------------------------------------------------------------------------------------------------------------------


# --------  Validate the results of the best individual of the population, not the entired population  -----------------
def validation_genome(genome):
    counter02 = 0
    fit_Val = [0 for n in xrange(0, len(validation_set[0]))]# List for store the fitness of each validation position
                                                            # en cada posicion de validacion.
    for pos in xrange(0, len(validation_set[0])):           # Loop for validate the best individual

        Y_values = [0]
        Y_values[0] = validation_set[0][pos][0]

        fit_Val[counter02], distance = ev_single_genome(genome, Y_values)
        counter02 = cp.deepcopy(counter02 + 1)

    return np.average(fit_Val)                              # Returned fitness value
# ----------------------------------------------------------------------------------------------------------------------


# -------------  Testing the best genome of the evolution  -------------------------------------------------------------
def test_genome(winner):
    counter02 = 0
    fit_Test = [0 for n in xrange(0, len(test_set[0]))] # List for store the fitness values.
    distance = [0 for n in xrange(0, len(test_set[0]))] # List for store the difference between the expected value, and
                                                        # the value given by the NN of the NEAT
    for test in xrange(0, len(test_set[0])):            # Loop for testing the best individuo in the test positions.
        Y_values = [0]
        Y_values[0] = test_set[0][test][0]

        fit_Test[counter02], distance[counter02] = ev_single_genome(winner, Y_values)
        counter02 = cp.deepcopy(counter02 + 1)

    return distance, fit_Test                           # Return the fitness and distance values of the test data
# ----------------------------------------------------------------------------------------------------------------------


# ---------------  Function for evolve the genome. Each genome is evaluated for each "x" values of the array  ----------
def train_genome(genoma):
    counter02 = 0
    a_fit_ens = [0 for n in xrange(0, len(train_set[0]))]# Lista para almacenar los valores de fitness de cada individuo
                                                         # en cada posicion de entrenamiento.
    for pos in xrange(0, len(train_set[0])):

        Y_values = [0]
        Y_values[0] = train_set[0][pos][0]

        a_fit_ens[counter02], distancia = ev_single_genome(genoma, Y_values)
        counter02 = cp.deepcopy(counter02 + 1)

    return np.average(a_fit_ens)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def run_population_evolution():

    genoma = NEAT.Genome(0, 2, num_neuron_hidden, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, seed_gen, params, num_hidden_layers)
    pop = NEAT.Population(genoma, params, True, 1.0, 0)

    best_index_ever = 0
    best_generation_ever = 0

    for generation in xrange(0, full_generations):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, train_genome, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)

        best = max(fitness_list)
        best_index = fitness_list.index(max(fitness_list))
        datah.evhist.append(best)
        if best > datah.best_fit_ever:
            sys.stdout.flush()
            datah.best_gs.append(pop.GetBestGenome())
            datah.best_fit_ever = best
            best_index_ever = best_index
            best_generation_ever = generation
        else:
            pass


        # Store the species contained in the population
        storeSpecies = []
        numSpecies = [0 for n in xrange(0, len(pop.Species))]
        for n in xrange(0, len(numSpecies)):
            numSpecies[n] = specie()
            numSpecies[n].ID = pop.Species[n].ID()
            numSpecies[n].tamano = pop.Species[n].NumIndividuals()
            storeSpecies.append(numSpecies[n])
        datah.species.append(storeSpecies)

        # Store the statistical data for representation of the evolution
        datah.bestIndividualsInGeneration[datah.generation] = pop.GetBestGenome()
        datah.valueFitnessMax[datah.generation] = best
        datah.valueFitnessMediana[datah.generation] = np.median(fitness_list)
        datah.perc75[datah.generation] = np.percentile(fitness_list, 75)
        datah.perc25[datah.generation] = np.percentile(fitness_list, 25)
        datah.valueFitnessMedia[datah.generation] = np.average(fitness_list)
        datah.fitnessValidation[datah.generation] = validation_genome(pop.GetBestGenome())
        datah.generation = cp.deepcopy(datah.generation + 1)

        if datah.generation > datah.max_generations - 1:
            sf.saveStatistics(datah, directory)
        else:
            pass

        if datah.generation > datah.max_generations - 1:
            sf.representationTrain_Validation(datah, directory, topLimit)
        else:
            pass

        pop.Epoch()
        print 'Best_fit_ever:', datah.best_fit_ever
        print('Generacion:', generation, 'Mejor:', round(best, 3), 'Mediana:', round(np.median(fitness_list), 3), 'Media:',round(np.average(fitness_list), 3), 'Std:', round(np.std(fitness_list), 3))
        print(' ')

    return (best_generation_ever, best_index_ever, datah.best_gs[len(datah.best_gs) -1], datah.best_fit_ever)


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def main():
    experiments = 1

    for expe in range(0, experiments):
        rd.seed(expe)

        datah.evhist = []
        datah.best_gs = []
        datah.dev_stage = 3
        datah.generation = 0

        best_individual = run_population_evolution()    # Evolution process
        genoma = best_individual[2]

        winner_genome = genoma
        net = NEAT.NeuralNetwork()
        winner_genome.BuildPhenotype(net)

        counter = 0
        for conn in net.connections:
            info = str(counter) + ' Conexiones: ' + '(' + str(conn.source_neuron_idx) + ', ' + str(
                conn.target_neuron_idx) + ') ' + '- weight: ' + str(conn.weight)
            counter += 1
            print info
        print ''


        print (' ')
        print('Best genome: ', winner_genome.GetID())
        print('Number of conections:', len(net.connections))
        print('Number of neurons;', len(net.neurons))
        print('Best Generation: ', best_individual[0])
        print('Best Individual: ', best_individual[1])
        print('Best fitness: ', round(best_individual[3], 3))
        print(' ')

        # TEST part
        distancia, fit_Test = test_genome(winner_genome)



        print 'Minimum fitness (mm):' + str(min(fit_Test))
        print 'Average distance (mm):' + str(round(np.average(distancia), 2))
        print 'Median distance (mm):' + str(round(np.median(distancia), 2))
        print 'Average fitness' + str(round(np.average(fit_Test), 2))
        porcentaje_media = round(np.average(distancia) * 100.0 / topLimit, 2)
        porcentaje_mediana = round(np.median(distancia) * 100.0 / topLimit, 2)
        print '% average distance:' + str(porcentaje_media)
        print '% median distance:' + str(porcentaje_mediana)

        text = directory + '/' + 'median_distance.png'
        sf.representationTest(distancia, text, porcentaje_media, porcentaje_mediana, topLimit)

        # Sspecies representation
        id_array = []
        for n in xrange(0, len(datah.species)):
            for m in xrange(0, len(datah.species[n])):
                id_array.append(datah.species[n][m].ID)

        SpeciesMaxNumber = max(id_array)
        format00 = [0 for n in range(0, SpeciesMaxNumber)]
        format01 = [cp.deepcopy(format00) for n in xrange(0, len(datah.species))]

        for indice in xrange(0, datah.generation):
            for num_especie in xrange(0, len(datah.species[indice])):
                format01[indice][datah.species[indice][num_especie].ID - 1] = datah.species[indice][num_especie].tamano

        species_sizes = format01
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(xrange(num_generations), *curves)

        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        text = directory + '/' + 'Species.png'
        plt.savefig(text)
        plt.close()
        # End of the stack of "species representation"

        # Drawing the net
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        NEAT.DrawPhenotype(img, (0, 0, 500, 500), net)
        text = directory + '/' + 'Best_Net.png'
        cv2.imwrite(text, img)


if __name__ == '__main__':
    main()