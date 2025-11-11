import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use('TkAgg')
import numpy as np
import matplotlib.ticker as ticker
import cv2


font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}


font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


def pain_seg_seismic_data(para_seismic_data, output_filename):
    """
    Plotting seismic data images of SEG salt datasets
    :param para_seismic_data:  Seismic data (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8.1), dpi = 120)
    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels = [0,0.75,1.5,2.25,3.0], size=16)
    ax.set_yticklabels(labels = [0.0,0.50,1.00,1.50,2.00], size=16)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)
    plt.savefig(output_filename)
    plt.show()


def pain_openfwi_seismic_data(para_seismic_data, output_filename):
    """
    Plotting seismic data images of openfwi dataset
    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    """
    # —— 1. 生成与数据同形状、同 dtype 的噪声 ——
    # noise = np.random.normal(loc=0.0, scale=0.3,
    #                          size=para_seismic_data.shape).astype(para_seismic_data.dtype)
    #
    # # —— 2. 叠加噪声 ——
    # para_seismic_data = para_seismic_data + noise  # 此时数据已带噪

    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.2, 8.1), dpi = 120)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=16)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=16)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)
    plt.savefig(output_filename)  # 保存图像
    plt.show()
    plt.close()


def plot_ground_truth(num, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(target, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'GT' + str(num) + '.png')
    plt.close(fig)


def plot_seg_prediction_velocity(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=150)
    im = ax.matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(0, 201, 25))
    ax.set_xticklabels(labels=[0.0,0.5,1.0,1.5,2.0,2.5,3.0], size=16)
    ax.set_yticklabels(labels=[0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00], size=16)
    ax.set_xlabel('Position (km)', size=16)
    ax.set_ylabel('Depth (km)', size=16)
    plt.rcParams['font.size'] = 14
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'PD' + str(num))  # 设置保存名字
    plt.close('all')


def plot_seg_truth_velocity(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=150)
    im = ax.matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(0, 201, 25))
    ax.set_xticklabels(labels=[0.0,0.5,1.0,1.5,2.0,2.5,3.0], size=16)
    ax.set_yticklabels(labels=[0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00], size=16)
    ax.set_xlabel('Position (km)', size=16)
    ax.set_ylabel('Depth (km)', size=16)
    plt.rcParams['font.size'] = 14
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'GT' + str(num))  # 设置保存名字
    plt.close('all')


def plot_prediction(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(output, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'PD' + str(num) + '.png')
    plt.close(fig)


def pain_openfwi_velocity_model(para_velocity_model):
    """
    Plotting seismic data images of openfwi dataset
    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)

    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.35)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.savefig('curveVelB.png')
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
    plt.show()


if __name__ == '__main__':
    # seismic_flatfaultA = np.load("G:/Data/OpenFWI/FlatFaultA/test_data/seismic/seismic1.npy")
    # seismic_curvefaultA = np.load("G:/Data/OpenFWI/CurveFaultA/test_data/seismic/seismic1.npy")
    # seismic_curvevelA = np.load("G:/Data/OpenFWI/CurveVelA/test_data/seismic/seismic1.npy")
    seismic_curvevelB = np.load("D:/Data/OpenFWI/CurveVelB/test_data/seismic/seismic1.npy")
    # seismic_curvefaultB = np.load("D:/Data/OpenFWI/CurveFaultB/test_data/seismic/seismic1.npy")
    # seismic_flatvelA = np.load("G:/Data/OpenFWI/FlatVelA/test_data/seismic/seismic1.npy")
    vmodel_curveVelB = np.load("D:/Data/OpenFWI/CurveVelB/test_data/vmodel/model1.npy")
    # pain_openfwi_velocity_model(vmodel_curveVelB[71, 0, :, :])

    pain_openfwi_seismic_data(seismic_curvevelB[71, 2, :, :], output_filename='noise.png')

    # seismic_SEGSimulation = scipy.io.loadmat("G:/Data/SEG/SEGSimulation/train_data/seismic/seismic1381.mat")["Rec"]
    # seismic_SEGSalt = scipy.io.loadmat("G:/Data/SEG/SEGSaltData/train_data/seismic/seismic108.mat")["Rec"]

    # pain_openfwi_seismic_data(seismic_flatflautA[3, 2, :, :])
    # pain_openfwi_seismic_data(seismic_flatflautA[10, 2, :, :])

    # pain_openfwi_seismic_data(seismic_curvevelA[190, 2, :, :], output_filename='Seismic_Data_CurveVelA.png')
    # pain_openfwi_seismic_data_noise(seismic_curvevelB[16, 4, :, :], output_filename='Seismic_Data_CurveVelB_noise4.png')
    # pain_openfwi_seismic_data(seismic_curvefaultA[106, 2, :, :], output_filename='Seismic_Data_CurveFaultA.png')
    # pain_openfwi_seismic_data(seismic_curvefaultB[14, 2, :, :], output_filename='Seismic_Data_CurveFaultB.png')
    # pain_openfwi_seismic_data(seismic_flatfaultA[28, 2, :, :], output_filename='5.png')
    # pain_openfwi_seismic_data(seismic_flatvelA[4, 2, :, :], output_filename='Seismic_Data_FlatVelA.png')

    # pain_seg_seismic_data(seismic_SEGSimulation[:, :, 15], output_filename='Seismic_Data_SEGSimulation.png')
    # pain_seg_seismic_data(seismic_SEGSalt[:, :, 15], output_filename='Seismic_Data_SEGSalt.png')
