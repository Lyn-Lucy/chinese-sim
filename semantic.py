# 先安装sentence_transformers
from sentence_transformers import SentenceTransformer

# Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased')
# distiluse-base-multilingual-cased 蒸馏得到的，官方预训练好的模型

# 加载数据集
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            try:
                text1, text2, label = l.strip().split(',')
                D.append((text1, text2, int(label)))
            except ValueError:
                continue
    return D

train_data = load_data('./input/train.csv')
valid_data = load_data('./input/dev.csv')
test_data  = load_data('./input/test.csv')

from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers import InputExample, evaluation, losses
from torch.utils.data import DataLoader

# Define your train examples.
train_datas = []
for i in train_data:
    train_datas.append(InputExample(texts=[i[0], i[1]], label=float(i[2])))

# Define your evaluation examples
sentences1,sentences2,scores = [],[],[]
for i in valid_data:
    sentences1.append(i[0])
    sentences2.append(i[1])
    scores.append(float(i[2]))

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)


# Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_datas, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

print("begin")
# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100,
          evaluator=evaluator, evaluation_steps=200, output_path='./two_albert_similarity_model')
