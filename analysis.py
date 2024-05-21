from argparse import ArgumentParser
from collections import OrderedDict
import pandas as pd
import json
import os

RESULT_ROOT = './analysis_results'

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--recover-time", action='store_true')
    parser.add_argument("--column-name", type=str, default=None)

    args = parser.parse_args()

    try:
        result = pd.read_csv(os.path.join(args.input_dir, 'train.csv'))
        params = json.load(open(os.path.join(args.input_dir, 'params.json')))
        data_config = json.load(open(params['data_config']))
    except FileNotFoundError:
        print("Invalid input directory:", args.input_dir)
        exit(1)

    result = result.rename(columns={col: col.strip(' ') for col in result.columns})

    if args.column_name is None:
        column_name = params['defender'] if params['defender'] is not None else 'No Def.'
    else:
        column_name = args.column_name
    
    if args.output_name is None:
        output_name = f"{params['model']}_{data_config['dataset']}_{params['local_epochs']}_{params['batch_size']}_{params['num_rounds']}"
    else:
        output_name = args.output_name

    output_dir = os.path.join(RESULT_ROOT, output_name)

    os.makedirs(output_dir, exist_ok=True)

    try:
        accuracy_df = pd.read_csv(os.path.join(output_dir, 'accuracy.csv'))
        table = json.load(open(os.path.join(output_dir, 'table.json')), object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        accuracy_df = pd.DataFrame(columns=['Round'], data=range(1, params['num_rounds']+1))
        table = {}

    if column_name not in accuracy_df.columns:
        accuracy_df[column_name] = result['Test Acc']

    if args.recover_time:
        table[column_name]['execute_time'] = result['Cumulative Time'].iloc[-1] / params['num_rounds']
    elif column_name not in table:
        table[column_name] = {
            'Accuracy': result['Test Acc'].iloc[-1],
            'last5_Accuracy_mean': result['Test Acc'].iloc[-5:].mean(),
            'last5_Accuracy_max': result['Test Acc'].iloc[-5:].max(),
            'last5_Accuracy_median': result['Test Acc'].iloc[-5:].median(),
            'last10_Accuracy_mean': result['Test Acc'].iloc[-10:].mean(),
            'last10_Accuracy_max': result['Test Acc'].iloc[-10:].max(),
            'last10_Accuracy_median': result['Test Acc'].iloc[-10:].median(),
            'execute_time': result['Cumulative Time'].iloc[-1] / params['num_rounds'],
        }
    
    accuracy_df.to_csv(os.path.join(output_dir, 'accuracy.csv'), index=False)
    json.dump(table, open(os.path.join(output_dir, 'table.json'), 'w'), indent=4)

    # print data
    with open(os.path.join(output_dir, 'output'), 'w') as file:
        file.write(f'{params['model']} – {data_config['dataset']} – batch size {params['local_epochs']} - local epoch {params['batch_size']} – {params['num_rounds']} round\n\n')

        file.write('\t'.join(table.keys()) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['Accuracy'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last5_Accuracy_mean'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last5_Accuracy_max'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last5_Accuracy_median'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last10_Accuracy_mean'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last10_Accuracy_max'] * 100), table.values())) + '\n')
        file.write('\t'.join(map(lambda data: "%.2f%%" % (data['last10_Accuracy_median'] * 100), table.values())) + '\n')

        first_time = next(iter(table.values()))['execute_time']
        file.write('\t'.join(map(
            lambda item: '%.2f（%+.2f%%）' % (item[1]['execute_time'], (item[1]['execute_time'] - first_time) / first_time * 100) if item[0] > 0 else '%.2f' % item[1]['execute_time'],
            enumerate(table.values())
        )) + '\n')

        
        
