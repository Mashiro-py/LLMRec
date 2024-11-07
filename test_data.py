#%%
import pandas as pd
import random

def reduce_database(input_file, output_file, sample_ratio=0.1, min_students=5):
    # 读取CSV文件
    df = pd.read_csv(input_file,encoding='GBK')
    
    # 获取唯一的学生ID
    student_ids = df['stu_id'].unique()
    
    # 确定要保留的学生数量
    num_students_to_keep = max(int(len(student_ids) * sample_ratio), min_students)
    
    # 随机选择要保留的学生ID
    selected_students = random.sample(list(student_ids), num_students_to_keep)
    
    # 筛选出选中学生的数据
    reduced_df = df[df['stu_id'].isin(selected_students)]
    
    # 重置course_index
    reduced_df['course_index'] = range(len(reduced_df))
    
    # 保存结果到新的CSV文件
    reduced_df.to_csv(output_file, index=False)
    
    print(f"原始数据中的学生数量: {len(student_ids)}")
    print(f"删减后的学生数量: {len(selected_students)}")
    print(f"原始数据中的记录数量: {len(df)}")
    print(f"删减后的记录数量: {len(reduced_df)}")

# 使用示例
input_file = 'data/mooc_data/data.csv'  # 请替换为您的输入文件名
output_file = 'data/mooc_data/data2.csv'  # 输出文件名
reduce_database(input_file, output_file, sample_ratio=0.1, min_students=5)