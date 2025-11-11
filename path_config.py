import os
from param_config import *

###################################################
####                DATA   PATHS              #####
###################################################

if OpenFWI == True:
    Data_path = 'G:/Data/OpenFWI/' + DataSet
elif SEGSaltData == True:
    Data_path = 'G:/Data/SEG/' + DataSet
elif SEGSimulateData == True:
    Data_path = 'G:/Data/SEG/' + DataSet
elif Marmousi == True:
    Data_path = 'D:/Data/Marmousi/' + DataSet


###################################################
####            RESULT   PATHS                #####
###################################################
main_dir = 'G:/Wang-Yumei/AMFMS-main/'

# Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

# Save training result
if os.path.exists('train_result/' + DataSet):
    train_result_dir = main_dir + 'train_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('train_result/' + DataSet)
    train_result_dir = main_dir + 'train_result/' + DataSet

# Save testing result
if os.path.exists('test_result/' + DataSet):
    test_result_dir = main_dir + 'test_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('test_result/' + DataSet)
    test_result_dir = main_dir + 'test_result/' + DataSet

####################################################
####                   FileName                #####
####################################################
# Set model name for training
saveModelName = 'AMFMS'

# U-Net_DEBW_MFR-HAR_MMT

tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

ModelName = saveModelName + tagM1 + tagM2 + tagM3 + tagM4

TestModelName = 'CurveVelA_InversionNet'

# Load the pre-trained model
PreModelname = TestModelName + '.pkl'
