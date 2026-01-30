####################################################
####             MAIN PARAMETERS                ####
####################################################

# choose the dataset
SEGSimulateData = False # True
SEGSaltData = False
OpenFWI = True
Marmousi = False
DataSet = "CurveVelA/"  # CurveVelA/B  FlatFaultA  FlatVelA   CurveFaultA/B  SEGSaltData  SEGSimulation

# choose the network
NetworkName = 'AMFMS'  # DD_Net DD_Net70 FCNVMB InversionNet AMFMS AMFMS_SEG ABA-FWI ABA-FWI+ VelocityGAN TU-Net TU-Net-SEG

# whether to use and train the model
ReUse = False  # If False always re-train a network

if OpenFWI or Marmousi:
    DataDim = [1000, 70]  # Dimension of original one-shot seismic data
    ModelDim = [70, 70]  # Dimension of one velocity model
    InChannel = 5  # Source numbers
    OutChannel = 1  # Number of channels in the output velocity model

elif SEGSaltData or SEGSimulateData:
    DataDim = [400, 301]  # Dimension of original one-shot seismic data
    ModelDim = [201, 301]  # Dimension of one velocity model
    InChannel = 29  # Source numbers
    OutChannel = 1  # Number of channels in the output velocity model

dh = 10  # Space interval

####################################################
####             NETWORK PARAMETERS             ####
####################################################

if NetworkName == "InversionNet" or NetworkName == "ABA-FWI":
    LearnRate = 1e-4
elif NetworkName in ["AMFMS", "AMFMS_SEG"]:
    LearnRate = 5e-4
elif NetworkName == "TU-Net" or NetworkName == "TU-Net-SEG":
    LearnRate = 3e-4
else:
    LearnRate = 1e-3


if DataSet == "FlatVelA/" or DataSet == "CurveVelA/" or DataSet == "CurveVelB/":
    Epochs = 140
    TrainSize = 24000  # 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 10
    BatchSize = 20  # Number of batch size
    SaveEpoch = 10
    loss_weight = [1, 0.01]
elif DataSet == "FlatFaultA/" or DataSet == "CurveFaultA/" or DataSet == "CurveFaultB/":
    Epochs = 140
    TrainSize = 48000  # 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 10
    BatchSize = 20  # Number of batch size
    SaveEpoch = 10
    loss_weight = [1, 0.01]
elif DataSet == "SEGSaltData/":
    Epochs = 60
    TrainSize = 130
    ValSize = 2
    TestSize = 10
    TestBatchSize = 1
    BatchSize = 10  # Number of batch size
    SaveEpoch = 10
    loss_weight = [1, 1e6]
elif DataSet == "SEGSimulation/":
    Epochs = 140
    TrainSize = 1600  # 1600
    ValSize = 10
    TestSize = 100
    TestBatchSize = 2
    BatchSize = 10  # Number of batch size
    SaveEpoch = 10
    loss_weight = [1, 1e6]
elif DataSet == "marmousi_70_70/":
    Epochs = 160
    TrainSize = 30926  # 30926
    ValSize = 500
    TestSize = 328
    TestBatchSize = 1
    BatchSize = 20  # Number of batch size
    SaveEpoch = 10
    loss_weight = [1, 0.01]

