import time
from data.read_data import *
from data.loss import *
from data.utils import *
from model.AMFMS import *
from math import cos, pi

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if NetworkName == "AMFMS":
    net = AMFMS(n_classes=OutChannel,
                     in_channels=InChannel,
                     is_deconv=True,
                     is_batchnorm=True)

elif NetworkName == "AMFMS_SEG":
    net = AMFMS_SEG(n_classes=OutChannel,
                       in_channels=InChannel,
                       is_deconv=True,
                       is_batchnorm=True)


net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0.0005, lr_max=0.001, warmup_epoch = 14):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


lr_max = 0.001
lr_min = 0.0005
warmup_epoch = 14

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
valSet = Dataset_test_edge(Data_path, ValSize, 1, "seismic", "test")
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

train_loss_list = 0
val_loss_list = 0
learning_rates = [] # 初始化一个列表来存储学习率


def train():
    net.train()
    total_loss = 0
    TV_loss = 0
    loss12 = 0
    for i, (seismic_datas, velocity_models, edges) in enumerate(train_loader):
        net.train()
        seismic_datas = seismic_datas[0].to(device)  # Tensor:(20,5,1000,70)
        velocity_models = velocity_models[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)
        edges = edges[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)

        # Zero the gradient buffer
        optimizer.zero_grad()
        # Forward pass
        outputs = net(seismic_datas)

        if NetworkName in ["AMFMS"]:
            outputs = outputs.to(torch.float32)
            loss, base_loss, tv_loss = criterion_MMT_Open(outputs, velocity_models)
        elif NetworkName in ["AMFMS_SEG"]:
            outputs = outputs.to(torch.float32)
            loss, base_loss, tv_loss = criterion_MMT_SEG(outputs, velocity_models)
        else:
            raise ValueError(f"Unknown NetworkName: {NetworkName}")

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        TV_loss += tv_loss.item()
        loss12 += base_loss.item()
        loss = loss.to(torch.float32)  # Loss backward propagation
        loss.backward()
        optimizer.step()  # Optimize

    avg_loss = total_loss / len(train_loader)
    avg_TV_loss = TV_loss / len(train_loader)
    avg_loss12 = loss12 / len(train_loader)
    return avg_loss, avg_TV_loss, avg_loss12


def validate():
    total_loss = 0
    net.eval()
    TV_loss = 0
    loss12 = 0
    with torch.no_grad():
        for i, (seismic_datas, velocity_models, edges, vmodel_max_min) in enumerate(val_loader):

            seismic_datas = seismic_datas[0].to(device)
            velocity_models = velocity_models[0].to(device).to(torch.float32)
            edges = edges[0].to(device).to(torch.float32)
            optimizer.zero_grad()  # Zero the gradient buffer
            outputs = net(seismic_datas)

            if NetworkName in ["AMFMS"]:
                outputs = outputs.to(torch.float32)
                loss, base_loss, tv_loss = criterion_MMT_Open(outputs, velocity_models)
            elif NetworkName in ["AMFMS_SEG"]:
                outputs = outputs.to(torch.float32)
                loss, base_loss, tv_loss = criterion_MMT_SEG(outputs, velocity_models)
            else:
                raise ValueError(f"Unknown NetworkName: {NetworkName}")

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            total_loss += loss.item()
            loss12 += base_loss.item()
            TV_loss += tv_loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_TV_loss = TV_loss / len(val_loader)
        avg_loss12 = loss12 / len(val_loader)
        return avg_loss, avg_TV_loss, avg_loss12


for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    warmup_cosine(optimizer=optimizer, current_epoch=epoch, max_epoch=Epochs, lr_min=lr_min, lr_max=lr_max, warmup_epoch=warmup_epoch)
    train_loss, train_TV_loss, train_loss12 = train()
    val_loss, val_TV_loss, val_loss12 = validate()

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)  # 将当前学习率添加到列表中
    print(optimizer.param_groups[0]['lr'])

    # Show train and val loss every 1 epoch
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1},Train loss: {train_loss:.4f}, Val loss: {val_loss: .4f}")
        print(f"Train TV loss: {train_TV_loss:.4f}, Val TV loss: {val_TV_loss: .4f}")
        print(f"Train loss12: {train_loss12:.4f}, Val loss12: {val_loss12: .4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('-----------------------------------------------------------------------------------------')

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

SaveLearningRate(learning_rates=learning_rates,
                 SavePath=train_result_dir,
                 ModelName=ModelName)
