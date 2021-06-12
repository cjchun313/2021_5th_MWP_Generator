import numpy as np
import json



def write_true_json(y_true_que, y_true_eq, y_true_ans, write_path='train.json', total_len=100000):
    #y_true_que = y_true_que.astype(str).tolist()
    #y_true_eq = y_true_eq.astype(str).tolist()
    y_true_ans = np.squeeze(y_true_ans).astype(str).tolist()

    total_sample = dict()
    for i in range(total_len):
        sample = dict()
        sample['question'] = y_true_que[i]
        sample['equation'] = y_true_eq[i]
        sample['answer'] = y_true_ans[i]

        total_sample[str(i)] = sample

    with open(write_path, 'w') as outfile:
        json.dump(total_sample, outfile, ensure_ascii=False, indent="    ")

    print('true json file saved.')



def write_pred_json(y_pred_eq, y_pred_ans, write_path='pred.json', total_len=1000):
    #y_pred_que = y_pred_que.astype(str).tolist()
    #y_pred_eq = y_pred_eq.astype(str).tolist()
    y_pred_ans = np.squeeze(y_pred_ans).astype(str).tolist()

    total_sample = dict()
    for i in range(total_len):
        sample = dict()
        #sample['question'] = y_pred_que[i]
        sample['answer'] = y_pred_ans[i]
        sample['equation'] = y_pred_eq[i]

        total_sample[str(i)] = sample

    with open(write_path, 'w') as outfile:
        json.dump(total_sample, outfile, ensure_ascii=False, indent="    ")

    print('pred json file saved.')



def read_true_json(read_path):
    with open(read_path, encoding='euc-kr') as json_file:
        json_data = json.load(json_file)
        samples = len(json_data)

        que_data, eq_data, ans_data = [], [], []
        for i in range(samples):
            que_data.append(json_data[str(i)]['question'])
            eq_data.append(json_data[str(i)]['equation'])
            ans_data.append(json_data[str(i)]['answer'])

    print('true json file loaded.')

    return que_data, eq_data, ans_data



if __name__ == "__main__":
    que, eq, ans = read_true_json(read_path='test.json')

    questions = len(que)
    for i in range(questions):
        print(que[i], eq[i], ans[i])

        break
    #print(que.shape, eq.shape, ans.shape)

