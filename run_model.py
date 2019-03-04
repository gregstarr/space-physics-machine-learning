import glob
import mag_nn
import matplotlib.pyplot as plt
plt.style.use('ggplot')

params = {
    'mag_files': glob.glob("./data/mag_data_*.nc")[:3],
    'ss_file': "./data/substorms_2000_2018.csv",
    'data_interval': 96,
    'prediction_interval': 96,
    'val_size': 512,
    'batch_size': 64,
    'model_name': "Wider_Net"
}

net = mag_nn.ResMax(params)

TRAIN = True
if not TRAIN:
    net.load_checkpoint("./model/model.ckpt")
    stats = net.run_validation()
    print(stats)
else:
    TRAINING_EPOCHS = 3

    stats = net.train(TRAINING_EPOCHS)

    plt.figure()
    plt.title("Loss")
    plt.plot(stats['train_loss_hist'], label="Train")
    plt.plot(stats['val_loss_hist'], label="Validation")
    plt.legend()

    plt.figure()
    plt.title("Occurrence Accuracy")
    plt.plot(stats['train_accuracy_hist'], label="Train")
    plt.plot(stats['val_accuracy_hist'], label="Validation")
    plt.legend()

    plt.figure()
    plt.title("Time Error")
    plt.plot(stats['train_time_error_hist'], label="Train")
    plt.plot(stats['val_time_error_hist'], label="Validation")
    plt.legend()

    plt.figure()
    plt.title("Location Error")
    plt.plot(stats['train_loc_error_hist'][:, 0], label="Train - SMLT")
    plt.plot(stats['train_loc_error_hist'][:, 1], label="Train - CMLT")
    plt.plot(stats['train_loc_error_hist'][:, 2], label="Train - MLAT")
    plt.plot(stats['val_loc_error_hist'][:, 0], label="Val - SMLT")
    plt.plot(stats['val_loc_error_hist'][:, 1], label="Val - CMLT")
    plt.plot(stats['val_loc_error_hist'][:, 2], label="Val - MLAT")
    plt.legend()

    plt.show()
