# Author: Omar Mirza
# Date: 3/14/20
# Description: Takes a text file and preprocesses it making two files, padded.txt and preprocessed.txt

from collections import Counter


def removeElements(filename):
    lst = open(filename[:-4] + 'padded.txt',
               encoding="utf8").read().lower().split()
    k = 1
    counted = Counter(lst)
    with open(filename[:-4] + 'preprocessed.txt', 'x',
              encoding="utf8") as fileobject:
        # uses list comprehension to remove sparse words
        fileobject.writelines(
            str(el) + " " if counted[el] > k else ' <unk> ' for el in lst)
    return


def padSentence(filename):
    result = ""
    with open(filename, encoding="utf8") as fileobject:
        with open(filename[:-4] + 'padded.txt', 'x', encoding="utf8") as f:
            for line in fileobject:
                print(" <s> " + line.rstrip("\n") + " </s> ", file=f)
    return


def removeUnseenWords(testfile, trainingfile):
    traininglst = open(trainingfile[:-4] + 'preprocessed.txt',
                       'r',
                       encoding='utf8').read().lower().split()
    trainingcounted = Counter(traininglst)
    testlst = open(testfile[:-4] + 'padded.txt', 'r',
                   encoding='utf8').read().lower().split()
    testcounted = Counter(testlst)
    with open(testfile[:-4] + 'preprocessed.txt', 'x',
              encoding="utf8") as fileobject:
        # uses list comprehension to remove sparse words
        fileobject.writelines(
            str(el) + " " if trainingcounted[el] > 0 else ' <unk> '
            for el in testlst)  #
    return

# driver code
trainingfilepath = 'train.txt'
testfilepath = 'test.txt'

padSentence(trainingfilepath)
removeElements(trainingfilepath)

padSentence(testfilepath)
removeUnseenWords(testfilepath, trainingfilepath)