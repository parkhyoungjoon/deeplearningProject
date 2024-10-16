import pandas as pd
import os
model = SkinKitModel()
csv_file_path = './DATA/output_data_target.csv'
csv_file_path2 = './DATA/output_data_test.csv'
for features, targets in test_loader:
    data_array = model(features)
    df = pd.DataFrame(data_array.detach().numpy())
    # df = pd.DataFrame(target)
    if not os.path.exists(csv_file_path2):
        df.to_csv(csv_file_path2, index=False, header=True)
    else:
        df.to_csv(csv_file_path2, index=False, header=False, mode='a')
csv_file_path = './DATA/output_data_test_target.csv'
for features, targets in test_loader:
    # data_array = model(features)
    # df = pd.DataFrame(data_array.detach().numpy())
    df = pd.DataFrame(target)
    if not os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, index=False, header=True)
    else:
        df.to_csv(csv_file_path, index=False, header=False, mode='a')
TFDF = pd.read_csv('./DATA/output_data.csv')