train_x1, test_x1 = get_data('train_mouse', 'test_mouse', 6)

train_x2, train_y = data_cat('concatenate', 'train_mouse', '100_1')
test_x2, test_y = data_cat('concatenate', 'test_mouse', '100_1')
# data = np.load(f'../datasets/train_human/train_human_6mer.npz')
# train_x2, train_y = data['seq_vec'], data['label']
# data = np.load(f'../datasets/test_human/test_human_6mer.npz')
# test_x2, test_y = data['seq_vec'], data['label']

train3, test3 = np.load('../datasets/one-hot/train_mouse.npz'), np.load('../datasets/one-hot/test_mouse.npz')
train_x3, test_x3 = train3['seq'], test3['seq']

train_x1, train_x2, train_x3, train_y = torch.Tensor(train_x1), torch.Tensor(train_x2), torch.Tensor(train_x3), torch.Tensor(train_y)
test_x1, test_x2, test_x3, test_y = torch.Tensor(test_x1), torch.Tensor(test_x2), torch.Tensor(test_x3), torch.Tensor(test_y)

scaler = StandardScaler()
scaler = scaler.fit(train_x2)
train_x = scaler.transform(train_x2)
test_x = scaler.transform(test_x2)
train_x2 = torch.unsqueeze(train_x2, dim=1)
test_x2 = torch.unsqueeze(test_x2, dim=1)