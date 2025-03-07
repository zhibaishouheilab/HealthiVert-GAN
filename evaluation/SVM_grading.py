import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def evaluate_svm(filepath, features, output_txt='evaluation_results.txt'):
    # 加载数据
    data = pd.read_excel(filepath)
    train_test_data = data[data['Dataset'].isin(['train', 'test'])]
    val_data = data[data['Dataset'] == 'val']

    # 准备输入和标签
    X_train_test = train_test_data[features]
    y_train_test = train_test_data['Label']
    X_val = val_data[features]
    y_val = val_data['Label']

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
    f1_list, precision_list, recall_list, accuracy_list = [], [], [], []

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
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)

    # 写入结果到文件
    with open(output_txt, 'w') as file:
        for i, (cm, f1, precision, recall, accuracy) in enumerate(results):
            file.write(f"Fold {i+1}:\n")
            file.write("Confusion Matrix:\n")
            file.write(f"{cm}\n")
            file.write(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")
            file.write("\n")
        
        # 计算平均分数和方差
        average_f1 = np.mean(f1_list)
        average_precision = np.mean(precision_list)
        average_recall = np.mean(recall_list)
        average_accuracy = np.mean(accuracy_list)
        variance_f1 = np.var(f1_list)
        variance_precision = np.var(precision_list)
        variance_recall = np.var(recall_list)
        variance_accuracy = np.var(accuracy_list)

        file.write("Average Scores:\n")
        file.write(f"Average F1 Score: {average_f1} (Variance: {variance_f1})\n")
        file.write(f"Average Precision: {average_precision} (Variance: {variance_precision})\n")
        file.write(f"Average Recall: {average_recall} (Variance: {variance_recall})\n")
        file.write(f"Average Accuracy: {average_accuracy} (Variance: {variance_accuracy})\n")

    print(f"Results saved to {output_txt}")

def main():
    result_folder = 'evaluation/RHLV_quantification'
    grading_folder = 'evaluation/classification_metric'
    if not os.path.exists(grading_folder):
        os.makedirs(grading_folder)
    features = ['Pre RHLV', 'Mid RHLV', 'Post RHLV']
    for xlsx_file in os.listdir(result_folder):
        #if xlsx_file != 'Exp_1_wo_straighten_sagittal.xlsx':
        #    continue
        xlsx_path = os.path.join(result_folder, xlsx_file)
        xlsx_name = xlsx_file.split('.')[0]
        saveTxT_path = os.path.join(grading_folder, xlsx_name + '.txt')
        evaluate_svm(xlsx_path, features, saveTxT_path)

if __name__ == "__main__":
    main()
