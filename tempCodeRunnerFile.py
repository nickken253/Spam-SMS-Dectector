if __name__ == "__main__":
    loaded_model = joblib.load('decision_tree_model.pkl')
    while(True):
        print("Nhap tin nhan: ")
        inp = input()
        print(demoDecisionTree(loaded_model, inp))