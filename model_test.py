import time
from data.read_data import *
from data.data_show import *
from model.AMFMS import *
from model.FCNVMB import *
from model.DDNet import *
from model.DDNet70 import *
from model.InversionNet import *
from model.ABA_FWI import *
from model.ABA_FWI_SEG import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
model_file = train_result_dir + PreModelname

if NetworkName == "AMFMS":
    net = AMFMS(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "AMFMS_SEG":
    net = AMFMS_SEG(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "InversionNet":
    net = InversionNet(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "DD_Net70":
    net = DDNet70Model(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "FCNVMB":
    net = FCNVMB(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "DD_Net":
    net = DDNetModel(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "ABA-FWI":
    net = ABA_FWI(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

elif NetworkName == "ABA-FWI+":
    net = ABA_FWI_SEG(n_classes=OutChannel, in_channels=InChannel, is_deconv=True, is_batchnorm=True)

net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** Loading dataset *****************')

dataset_dir = Data_path

testSet = Dataset_test_edge(dataset_dir, TestSize, 1, "seismic", "test")
test_loader = DataLoader(testSet, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                  Testing...               ')
print('*******************************************')
print('*******************************************')
print()

# Initialization
since = time.time()

Total_PSNR = np.zeros((1, TestSize), dtype=float)
Total_SSIM = np.zeros((1, TestSize), dtype=float)
Total_MSE = np.zeros((1, TestSize), dtype=float)
Total_MAE = np.zeros((1, TestSize), dtype=float)
Total_UQI = np.zeros((1, TestSize), dtype=float)
Total_LPIPS = np.zeros((1, TestSize), dtype=float)
Local_MSE = np.zeros((1, TestSize), dtype=float)
Local_MAE = np.zeros((1, TestSize), dtype=float)

Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
Prediction_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)

total = 0
for i, (seismic_datas, vmodels, edges, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    # Forward prediction
    outputs = net(seismic_datas)
    if NetworkName in ["DD_Net", "DD_Net70"]:
        outputs = outputs[0].data.cpu().numpy()
    else:
        outputs = outputs.data.cpu().numpy()

    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()

    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]

        contour_mask = extract_contours(gt)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(contour_mask, kernel, iterations=1)

        gt_vmodel_of_k = dilate * gt
        pd_vmodel_of_k = dilate * pd

        psnr = PSNR(gt, pd)
        ssim = SSIM(gt, pd)
        mse = MSE(pd, gt)
        mae = MAE(pd, gt)
        uqi = UQI(pd, gt)
        lpips = LPIPS(pd, gt)
        mse2 = MSE(pd_vmodel_of_k, gt_vmodel_of_k)
        mae2 = MAE(pd_vmodel_of_k, gt_vmodel_of_k)

        Total_PSNR[0, total] = psnr
        Total_SSIM[0, total] = ssim
        Total_MSE[0, total] = mse
        Total_MAE[0, total] = mae
        Total_UQI[0, total] = uqi
        Total_LPIPS[0, total] = lpips
        Local_MSE[0, total] = mse2
        Local_MAE[0, total] = mae2

        total = total + 1

        pd_N = pd * (vmax - vmin) + vmin
        gt_N = gt * (vmax - vmin) + vmin

        Prediction[i * TestBatchSize + k, :, :] = pd_N
        GT[i * TestBatchSize + k, :, :] = gt_N

        # if total in range(1, 100, 1):
        #     plot_ground_truth(total, gt_N, test_result_dir, vmin, vmax)  # Show ground truth image
        #     plot_prediction(total, pd_N, test_result_dir, vmin, vmax)  # Show prediction image

        # if total in range(1, 101, 1):
        #     plot_seg_prediction_velocity(total, pd, test_result_dir, vmin, vmax)
        #     plot_seg_truth_velocity(total, gt, test_result_dir, vmin, vmax)


        print('The %d testing psnr: %.2f, SSIM: %.4f, MSE: %.4f, MAE: %.4f, UQI: %.4f, LPIPS: %.4f, '
              'Local_MSE: %.4f, Local_MAE: %.4f' % (total, psnr, ssim, mse, mae, uqi, lpips, mse2, mae2))

SaveTestResultsEdge(Total_PSNR, Total_SSIM, Total_MSE, Total_MAE, Total_UQI, Total_LPIPS, Local_MSE, Local_MAE,
                    Prediction, GT, test_result_dir)

