"""
train the model
- load the training data(json file)
- split it into x(pattern) and y(tags)
- translate x(pattern) to bag_of_words
- index y(tags)
- start training (deep learning)
- save the model

start chat bot
- get input from user
-- if input == "quit", quit chat bot; if input tag == "teach", start teaching; else do the following
- translate it to bag_of_words
- use the trained model (and fit the input) to predict y(tag)
-- try to use a saved model, else train a new one
- use the y(tag) to give response
"""
import nltk
import numpy as np
import tflearn
import random
import json


stemmer = nltk.stem.lancaster.LancasterStemmer()
all_words = []
tags = []
with open("intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = nltk.word_tokenize(pattern)
        all_words.extend(words)

    if intent["tag"] not in tags:
        tags.append(intent["tag"])

all_words = [stemmer.stem(word.lower()) for word in all_words if word != "?"]
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

"""
# tran_bag
parameter: (["How are you doing?"], ["how", "a", "b", "are", "you", "do"])
["How are you doing?"]
-> split it, lower case, ignore symbols, stem
-> ["how", "are", "you", "do"]
-> return: [1, 0, 0, 1, 1, 1]
"""


def tran_bag(sentence, all_words_list):
    bag = [0 for _ in range(len(all_words_list))]
    word_tokens = nltk.word_tokenize(sentence)
    word_tokens = [stemmer.stem(word.lower()) for word in word_tokens if word != "?"]
    for word in all_words_list:
        for word_token in word_tokens:
            if word_token == word:
                bag[all_words_list.index(word_token)] = 1
    return bag


"""
# train
-> x = [[0,1,1,0], [1,0,0,0], [0,0,0,1]] (bag of words)
-> y = [[0,1,0], [1,0,0], [0,0,1]] (one-hot)
make a model
fit x,y into the model
save the model
"""


def train(train_bool=False):
    x = []
    y = []
    y_empty = [0 for _ in range(len(tags))]
    for i in data["intents"]:
        for p in i["patterns"]:
            bag = tran_bag(p, all_words)
            x.append(bag)
            y_row = y_empty[:]
            y_row[tags.index(i["tag"])] = 1
            y.append(y_row)

    x = np.array(x)
    y = np.array(y)

    net = tflearn.input_data(shape=[None, len(x[0])])  # len(x[0]) = 32
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(y[0]), activation="softmax")  # len(x[0]) = 6
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    if train_bool:
        model.fit(x, y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
        print("Chat bot: " + "I've finished learning!")
    else:
        model.load("model.tflearn")

    return model


def chat():
    start_loop = True
    while start_loop:
        print("Chat bot: " + "Train me first? (type yes or no)")
        inp_train = input("You: ")
        if inp_train.lower() == "yes":
            train_bool = True
            start_loop = False
        elif inp_train.lower() == "no":
            train_bool = False
            start_loop = False
        else:
            print("Chat bot: " + "I don't know what you mean! Try again")

    model = train(train_bool)
    print("Start talking with the bot!")
    while True:
        # get user input
        inp = input("You: ")

        # translate the input and make a prediction
        input_bags = np.array(tran_bag(inp, all_words))
        results = model.predict([input_bags])
        results_index = np.argmax(results)
        tag = tags[results_index]

        # give response
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print("Chat bot: " + random.choice(responses))

        # quit or teach
        if tag == "quit":
            break
        elif tag == "teach":
            # get teaching topic(tag)
            inp_teach = input("You: ")

            # find the topic(tag)
            input_bags_teach = np.array(tran_bag(inp_teach, all_words))
            results_teach = model.predict([input_bags_teach])
            results_index_teach = np.argmax(results_teach)
            tag_teach = tags[results_index_teach]

            # confirm and get response
            print("Chat bot: " + "So you want to teach me about " + tag_teach + "? Tell me how to response to that topic!")
            inp_response = input("You: ")
            print("Chat bot: " + "OK, I am trying to memorise it!")
            for tg in data["intents"]:
                if tg['tag'] == tag_teach:
                    tg["responses"].append(inp_response)
            # rewrite json file
            with open('intents.json', 'w') as outfile:
                json.dump(data, outfile)


chat()
