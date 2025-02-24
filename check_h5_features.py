def read_csv(path):
    with open(path, 'r') as f:
        return f.readlines()

def __main__():
    h5_features_actuals = read_csv('output.csv')
    h5_features_preds = read_csv('h5_features_preds.csv')
    for line in lines:
        print(line)
