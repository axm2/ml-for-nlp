import math
from collections import Counter
from itertools import islice

trainpreprocessedpath = "trainpreprocessed.txt"
testpreprocessedpath = "testpreprocessed.txt"
traincorpus = open(trainpreprocessedpath, encoding="utf8").read().split()
traincount = Counter(traincorpus)  # unigrams for the training corpus
testcorpus = open(testpreprocessedpath, encoding="utf8").read().split()
testcount = Counter(testcorpus)  # unigrams for the test corpus
bigramtraincount = Counter(zip(traincorpus, islice(traincorpus, 1, None)))
bigramtestcount = Counter(zip(testcorpus, islice(testcorpus, 1, None)))

# question 1
print("Question 1: ")
print("Number of word types in training corpus: " + str(len(traincount)))
# question 2
print("Question 2:")
print("Number of word tokens in training corpus: " + str(sum(traincount.values())))
# question 3
print("Question 3: ")
percentageunktypes = 1 / len(testcount) * 100
percentageunktokens = testcount["<unk>"] / sum(testcount.values()) * 100
print(
    "Percentage of word types in the test corpus that did not occur in training: "
    + str(percentageunktypes)
    + "%"
)
print(
    "Percentage of word tokens in the test corpus that did not occur in training: "
    + str(percentageunktokens)
    + "%"
)
# question 4
print("Question 4: ")
uniquebigramtypecounter = 0
for el in bigramtestcount:
    if bigramtraincount[el] == 0:
        uniquebigramtypecounter += 1
percentageuniquebigramtypecounter = uniquebigramtypecounter / len(bigramtestcount) * 100
print(
    "Percentage of bigram types that did not occur in training: "
    + str(percentageuniquebigramtypecounter)
)
uniquebigramtokencounter = 0
for el in bigramtestcount:
    if bigramtraincount[el] == 0:
        uniquebigramtokencounter += bigramtestcount[el]
percentageuniquebigramtokencounter = (
    uniquebigramtokencounter / sum(bigramtestcount.values()) * 100
)
print(
    "Percentage of bigram tokens that did not occur in training: "
    + str(percentageuniquebigramtokencounter)
)
# question 5
print("Question 5: ")
input = "I look forward to hearing your reply ."
# padding
input = "<s> " + input.lower() + " </s>"
# replace unseen words
inputsplit = input.split()
inputsplit = ["<unk>" if traincount[i] == 0 else i for i in inputsplit]
# now we calculate log probability under each model
unilogprob = 0
for el in inputsplit:
    unilogprob += math.log2(traincount[el] / sum(traincount.values()))
    print(
        "Log probability of "
        + el
        + " : "
        + str(math.log2(traincount[el] / sum(traincount.values())))
    )
print(
    "Unigram log probability of input: "
    + str(unilogprob)
    + " (summation of individual log probabilities)"
    + "\n"
)
# converting our input string to bigrams
bigraminputsplit = list(zip(inputsplit, inputsplit[1:]))
bilogprob = 0
for el in bigraminputsplit:
    if (bigramtraincount[el]) == 0:
        bilogprob = "undefined"
        print("Log probability of " + el[0] + " " + el[1] + " : undefined")
        # if we get a probability of undef, that means the whole sentence has probability of undefined
    else:
        if (
            bilogprob != "undefined"
        ):  # allows us to print individual probabilities even if we've encountered an undefined probability
            bilogprob += math.log2(bigramtraincount[el] / traincount[el[0]])
        print(
            "Log probability of "
            + el[0]
            + " "
            + el[1]
            + " : "
            + str(math.log2(bigramtraincount[el] / traincount[el[0]]))
        )
print(
    "Bigram log probability of input: "
    + str(bilogprob)
    + " (summation of individual log probabilities)"
    + "\n"
)
addonebilogprob = 0
# copy to get new iterator
addonebiloginputsplit = zip(inputsplit, inputsplit[1:])
for el in addonebiloginputsplit:
    addonebilogprob += math.log2(
        (bigramtraincount[el] + 1) / (traincount[el[0]] + len(traincount))
    )
    print(
        "Add-one Log probability of "
        + el[0]
        + " "
        + el[1]
        + " : "
        + str(
            math.log2(
                (bigramtraincount[el] + 1) / (traincount[el[0]] + len(traincount))
            )
        )
    )
print(
    "Add-one Bigram log probability of input: "
    + str(addonebilogprob)
    + " (summation of individual log probabilities)"
    + "\n"
)
# question 6
print("Question 6: ")
# first we compute the average log probabilities word by word
avgunilogprob = 0
avgbilogprob = 0
avgaddonebilogprob = 0
if unilogprob != "undefined":
    avgunilogprob = unilogprob / len(inputsplit)
else:
    avgunilogprob = "undefined"
if bilogprob != "undefined":
    avgbilogprob = bilogprob / len(inputsplit)
else:
    avgbilogprob = "undefined"
avgaddonebilogprob = addonebilogprob / len(inputsplit)
if avgunilogprob != "undefined":
    print(
        "perplexity of sentence under unigram model: "
        + str(math.pow(2, -1 * avgunilogprob))
    )
else:
    print("perplexity of sentence under unigram model is undefined")
if avgbilogprob != "undefined":
    print(
        "perplexity of sentence under bigram model: "
        + str(math.pow(2, -1 * avgbilogprob))
    )
else:
    print("perplexity of sentence under bigram model is undefined")
print(
    "perplexity of sentence under add-one bigram model: "
    + str(math.pow(2, -1 * avgaddonebilogprob))
)
# question 7
print("Question 7: ")
# first calculate the log probabilities for each model
testunilogprob = 0
for el in testcorpus:
    testunilogprob += math.log2(traincount[el] / sum(traincount.values()))
    # print(
    #     "Log probability of "
    #     + el
    #     + " : "
    #     + str(math.log2(testcount[el] / sum(testcount.values())))
    # )
# print(
#     "Unigram log probability of test: "
#     + str(testunilogprob)
#     + " (summation of individual log probabilities)"
#     + "\n"
# )
testbilogprob = 0
testbigraminputsplit = list(zip(testcorpus, islice(testcorpus, 1, None)))
for el in testbigraminputsplit:
    if (bigramtraincount[el]) == 0:
        testbilogprob = "undefined"
        # print("Log probability of " + el[0] + " " + el[1] + " : undefined")
    else:
        if testbilogprob != "undefined":
            testbilogprob += math.log2(bigramtraincount[el] / traincount[el[0]])
        # print(
        #     "Log probability of "
        #     + el[0]
        #     + " "
        #     + el[1]
        #     + " : "
        #     + str(math.log2(bigramtestcount[el] / testcount[el[0]]))
        # )
# print(
#     "Bigram log probability of test: "
#     + str(testbilogprob)
#     + " (summation of individual log probabilities)"
#     + "\n"
# )

testaddonebilogprob = 0
# copy to get new iterator
testaddonebiloginputsplit = list(zip(testcorpus, islice(testcorpus, 1, None)))
for el in testaddonebiloginputsplit:
    testaddonebilogprob += math.log2(
        (bigramtraincount[el] + 1) / (traincount[el[0]] + len(traincount))
    )
# print(
#     "Add-one Bigram log probability of test: "
#     + str(testaddonebilogprob)
#     + " (summation of individual log probabilities)"
#     + "\n"
# )

# now we compute the avg log probabilities

testavgunilogprob = 0
testavgbilogprob = 0
testavgaddonebilogprob = 0
if testunilogprob != "undefined":
    testavgunilogprob = testunilogprob / sum(testcount.values())
else:
    testavgunilogprob = "undefined"
if testbilogprob != "undefined":
    testavgbilogprob = testbilogprob / sum(testcount.values())
else:
    testavgbilogprob = "undefined"
testavgaddonebilogprob = testaddonebilogprob / sum(testcount.values())
if testavgunilogprob != "undefined":
    print(
        "perplexity of test under unigram model: "
        + str(math.pow(2, -1 * testavgunilogprob))
    )
else:
    print("perplexity of test under unigram model is undefined")
if testavgbilogprob != "undefined":
    print(
        "perplexity of test under unigram model: "
        + str(math.pow(2, -1 * testavgbilogprob))
    )
else:
    print("perplexity of test under unigram model is undefined")
print(
    "perplexity of test under add-one bigram model: "
    + str(math.pow(2, -1 * testavgaddonebilogprob))
)

