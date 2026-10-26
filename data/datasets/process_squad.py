from pathlib import Path
import argparse
import jsonlines


BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='处理 SQuAD 数据集，转换为 EoR 格式')
parser.add_argument('-d', '--data_path', type=str, default=str(BASE_DIR / 'squad_train.json'), help='path of data to processed')
parser.add_argument('-o', '--output_path', type=str, default=str(BASE_DIR / 'squad_train_processed.json'), help='path to save the processed data')


def extract_answer(item):
    if 'truthful answer' in item:
        return item['truthful answer']
    if 'answer' in item:
        return item['answer']
    answers = item.get('answers', {})
    if isinstance(answers, dict):
        texts = answers.get('text', [])
        if texts:
            return texts[0]
    return ''


args, _ = parser.parse_known_args()
print('开始处理数据...')
print(f'输入文件: {args.data_path}')
print(f'输出文件: {args.output_path}')

data = []
with jsonlines.open(args.data_path, mode='r') as reader:
    for line in reader:
        data.append(line)

print(f'读取了 {len(data)} 条数据')

processed_data = [
    {
        'query': item['question'],
        'truthful answer': extract_answer(item),
        'language': 'en',
    }
    for item in data
]

print('处理完成，准备保存...')

with jsonlines.open(args.output_path, mode='w') as writer:
    writer.write_all(processed_data)

print(f'✓ 成功保存到: {args.output_path}')
print(f'✓ 总共处理了 {len(processed_data)} 条数据')