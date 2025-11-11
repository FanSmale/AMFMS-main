import time
from data.read_data import *
from data.loss import *
from data.utils import *
from model.DDNet import *
from model.DDNet70 import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if NetworkName == "DD_Net70":
    net = DDNet70Model(n_classes=OutChannel,
                       in_channels=InChannel,
                       is_deconv=True,
                       is_batchnorm=True)

elif NetworkName == "DD_Net":
    net = DDNetModel(n_classes=OutChannel,
                     in_channels=InChannel,
                     is_deconv=True,
                     is_batchnorm=True)

net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodel_filepath and continue to train
if ReUse:
    print('***************** Loading pre-training model *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading training dataset *****************')


trainSet = Dataset_train_edge(Data_path, TrainSize, 1, "seismic", "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)
valSet = Dataset_test_edge(Data_path, ValSize, 1601, "seismic", "test")
val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                Training ...               ')
print('*******************************************')
print('*******************************************')
print()

print('原始地震数据尺寸:%s' % str(DataDim))
print('原始速度模型尺寸:%s' % str(ModelDim))
print('培训规模:%d' % int(TrainSize))
print('培训批次大小:%d' % int(BatchSize))
print('迭代轮数:%d' % int(Epochs))
print('学习率:%.5f' % float(LearnRate))

# Initialization
step = int(TrainSize / BatchSize)
start = time.time()


def train():
    net.train()
    total_loss = 0
    for i, (seismic_datas, velocity_models, edges) in enumerate(train_loader):
        net.train()
        seismic_datas = seismic_datas[0].to(device)  # Tensor:(20,5,1000,70)
        velocity_models = velocity_models[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)
        edges = edges[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)

        noise_mean = 0
        noise_std = 0.3
        noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
        seismic_datas = seismic_datas + noise

        # Zero the gradient buffer
        optimizer.zero_grad()
        # Forward pass
        outputs = net(seismic_datas)

        criterion_dd = LossDDNet(weights=loss_weight)
        outputs[0] = outputs[0].to(torch.float32)
        outputs[1] = outputs[1].to(torch.float32)
        loss = criterion_dd(outputs[0], outputs[1], velocity_models, edges)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        loss = loss.to(torch.float32)  # Loss backward propagation
        loss.backward()
        optimizer.step()  # Optimize

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate():
    total_loss = 0
    net.eval()
    with torch.no_grad():
        for i, (seismic_datas, velocity_models, edges, vmodel_max_min) in enumerate(val_loader):
            seismic_datas = seismic_datas[0].to(device)
            velocity_models = velocity_models[0].to(device).to(torch.float32)
            edges = edges[0].to(device).to(torch.float32)
            optimizer.zero_grad()  # Zero the gradient buffer
            outputs = net(seismic_datas)

            criterion_dd = LossDDNet(weights=loss_weight)
            outputs[0] = outputs[0].to(torch.float32)
            outputs[1] = outputs[1].to(torch.float32)
            loss = criterion_dd(outputs[0], outputs[1], velocity_models, edges)

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss


train_loss_list = 0
val_loss_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    train_loss = train()
    val_loss = validate()

    # Show train and val loss every 1 epoch
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1},Train loss: {train_loss:.4f}, Val loss: {val_loss: .4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    val_loss_list = np.append(val_loss_list, val_loss)

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

SaveTrainResults(train_loss=train_loss_list,
                 val_loss=val_loss_list,
                 SavePath=train_result_dir,
                 ModelName=ModelName,
                 font2=font2,
                 font3=font3)

