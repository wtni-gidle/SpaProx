import pickle
from model import nn_model
import evaluation as eval
import cupy as cp
import imb
import copy
if __name__ == "__main__":
    # 四个k得到的模型的各自的结果
    DIRNAME = "mouse_brain_sagittal_anterior"

    GPU_ID = 2

    k_candidate = range(5, 25, 5)

    with open(DIRNAME + "/train_data.pkl", "rb") as file:
        train_data = pickle.load(file)
    with open(DIRNAME + "/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    result_total = dict()

    test_data_cp = copy.deepcopy(test_data)
    with cp.cuda.Device(GPU_ID):
        pos_index, neg_index, marked_neg_index = imb.eliminate_BD_neg(test_data_cp.feature, test_data_cp.label, k = 20)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        marked_neg_index = cp.asnumpy(marked_neg_index)
    marked_feature = test_data_cp.get_feature(test_data_cp.pair_index_son[marked_neg_index], copy = True)
    marked_label = test_data_cp.get_label(test_data_cp.data_index[marked_neg_index], copy = True)
    test_data_cp.pop(marked_neg_index)
    test_data_cp.mirror_copy()
    test_data_cp.get_feature()
    test_data_cp.get_label()


    for k in k_candidate:
        train_data_cp = copy.deepcopy(train_data)
        with cp.cuda.Device(GPU_ID):
            pos_index, neg_index, marked_neg_index = imb.eliminate_BD_neg(train_data_cp.feature, train_data_cp.label, k = k)
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            marked_neg_index = cp.asnumpy(marked_neg_index)
        train_data_cp.pop(marked_neg_index)
        train_data_cp.mirror_copy()
        train_data_cp.get_feature()
        train_data_cp.get_label()
        model = nn_model.NeuralNetworkClassifier(batch_size = 128, verbose = False)
        model.fit(train_data_cp.feature, train_data_cp.label)

        predprob = model.predict_proba(test_data_cp.feature)
        result1 = eval.evaluate(test_data_cp.label, predprob, verbose = True)

        predprob = model.predict_proba(marked_feature)
        result2 = eval.evaluate(marked_label, predprob, verbose = True)

        result_total[str(k)] = [result1, result2]

    with open(DIRNAME + "/result2.pkl", "wb") as file:
        pickle.dump(result_total, file)

