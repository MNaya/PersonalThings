import MultiNEAT as NEATT

def parametersCreation():
    params = NEATT.Parameters()
    # -----------  Basic parameters  ---------

    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = True
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.9

    params.ArchiveEnforcement = False

    params.MutateWeightsProb = 0.05

    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 2.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.25
    params.WeightReplacementRate = 0.5

    params.MaxWeight = 10

    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.3
    params.MutateRemLinkProb = 0.0

    params.MinActivationA = 1.0
    params.MinActivationA = 1.0
    params.MaxActivationB = 1.0
    params.MaxActivationB = 1.0

    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0

    params.CrossoverRate = 0.1
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2

    params.MinNeuronBias = -params.MaxWeight
    params.MaxNeuronBias = params.MaxWeight

    return params

