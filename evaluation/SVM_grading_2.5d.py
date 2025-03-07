import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def evaluate_svm(file1, file2, features, output_txt='evaluation_results.txt'):
    # 加载数据
    data1 = pd.read_excel(file1)
    data2 = pd.read_excel(file2)

    # 重命名第二个文件的特征，以避免冲突
    rename_dict = {f: f"{f}_2" for f in features}
    data2.rename(columns=rename_dict, inplace=True)

    # 确保数据有一个共同的列来合并（例如 ID）
    combined_data = pd.merge(data1, data2, on="Vertebra")
    print(combined_data)

    # 选取参与训练和测试的数据
    train_test_data = combined_data[combined_data['Dataset_x'].isin(['train', 'test'])]
    val_data = combined_data[combined_data['Dataset_x'] == 'val']

    # 准备输入和标签
    combined_features = features + [f"{f}_2" for f in features]
    X_train_test = train_test_data[combined_features]
    y_train_test = train_test_data['Label_x']
    X_val = val_data[combined_features]
    y_val = val_data['Label_x']

    # 数据标准化
    scaler = StandardScaler()
    X_train_test_scaled = scaler.fit_transform(X_train_test)
    X_val_scaled = scaler.transform(X_val)

    # 初始化 SVM 分类器
    svm_classifier = SVC(kernel='linear', class_weight='balanced')

    # 设置五折交叉验证
    skf = StratifiedKFold(n_splits=5)

    # 存储每次验证的结果
    results = []

    for train_index, test_index in skf.split(X_train_test_scaled, y_train_test):
        X_train, X_test = X_train_test_scaled[train_index], X_train_test_scaled[test_index]
        y_train, y_test = y_train_test[train_index], y_train_test[test_index]
        
        svm_classifier.fit(X_train, y_train)
        y_pred_val = svm_classifier.predict(X_val_scaled)
        cm = confusion_matrix(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, average='macro')
        precision = precision_score(y_val, y_pred_val, average='macro')
        recall = recall_score(y_val, y_pred_val, average='macro')
        accuracy = accuracy_score(y_val, y_pred_val)
        
        results.append((cm, f1, precision, recall, accuracy))

    # 写入结果到文件
    with open(output_txt, 'w') as file:
        for i, (cm, f1, precision, recall, accuracy) in enumerate(results):
            file.write(f"Fold {i+1}:\n")
            file.write("Confusion Matrix:\n")
            file.write(f"{cm}\n")
            file.write(f"F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}\n")
            file.write("\n")
        
        # 计算平均分数
        average_f1 = np.mean([r[1] for r in results])
        average_precision = np.mean([r[2] for r in results])
        average_recall = np.mean([r[3] for r in results])
        average_accuracy = np.mean([r[4] for r in results])

        file.write("Average Scores:\n")
        file.write(f"Average F1 Score: {average_f1:.3f}\n")
        file.write(f"Average Precision: {average_precision:.3f}\n")
        file.write(f"Average Recall: {average_recall:.3f}\n")
        file.write(f"Average Accuracy: {average_accuracy:.3f}\n")

    print(f"Results saved to {output_txt}")

def main():
    result_folder = 'RHLV_quantification'
    grading_folder = 'classification_metric'
    if not os.path.exists(grading_folder):
        os.makedirs(grading_folder)
    
    file_1 = os.path.join(result_folder,'fine.xlsx')
    #file_1 = 'twostage_output.xlsx'
    file_2 = 'twostage_output.xlsx'
    features = ['Pre RHLV', 'Mid RHLV', 'Post RHLV']  # 特征

    output_txt_path = os.path.join(grading_folder, 'test.txt')
    evaluate_svm(file_1, file_2, features, output_txt_path)
    
if __name__ == "__main__":
    main()

