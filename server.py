from math import log2
from operator import index
import os
import sys
from unicodedata import name
from markupsafe import re
from tabulate import tabulate
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import flask
from flask import request
from flask import render_template
import json
from random import randint
from waitress import serve
import time
from pdfminer.high_level import extract_text
from guppy import hpy

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

port = '8000'

# Read arguments from shell
verbose = False
hideLow = False
memoryProfiling = False
amountOfThreads = 8
limit = 0
for arg in sys.argv:
    if arg == "-v":
        verbose = True
    if arg == "-u":
        hideLow = True
    if arg == "-mp":
        memoryProfiling = True
    if arg == "-l":
        try:
            limit = int(sys.argv[sys.argv.index("-l") + 1])
        except:
            print("Argument error: Please add [amount] after -l")
            exit()
    if arg == "-t":
        try:
            amountOfThreads = int(sys.argv[sys.argv.index("-t") + 1])
        except:
            print("Argument error: Please add [amount] after -t")
            exit()
    if arg == "-p":
        try:
            port = int(sys.argv[sys.argv.index("-p") + 1])
        except:
            print("Argument error: Please add [PORT] after -p")
            exit()

    if arg == "-h" or arg == "--help":
        print("==PaperFind Server==")
        print(" -h              Show this menu")
        print(" -p  [PORT]      Change server port. Default: 8000")
        print(" -v              Print all steps of calculation in terminal")
        print(" -u              Do not output ULTRA LOW results")
        print(" -l  [amount]    Limit the output to a certain amount. Default: 0")
        print(" -t  [amount]    Specify on how many threads the server should be active. Default: 8")
        print(" -mp             Profile memory use per query")
        exit()


# Start memory profyling
if memoryProfiling:
    h = hpy()

# Global runtime variables
frequencyTable = None
preIdf = None

# Making sure all required nltk packages are installed
print("Making sure all required nltk packages are installed")
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Simple logging function
def printLog(*input, table=False):
    if not table:
        for item in input:
            print(item, end=" ")
        print("\n", end="")
    if verbose and table:
        print("\n", input[0], "\n", end="")
        print(tabulate(input[1], headers="firstrow"))
        print("\n", end="")

# Function that reads text with support for multiple file extenstions
def readText(file):
    if file.name.split('.')[-1] == "txt":
        # Read text from file
        return file.read()
    elif file.name.split('.')[-1] == "pdf":
        # Read text from PDF
        return extract_text(file.name, 'rb')


# This function calculates the vector length per document id
def calculateVectorLength(TfIdf, documentColumn):
    squaredSum = 0
    for row in range(1, len(TfIdf)):
        squaredSum += TfIdf[row][documentColumn]**2
    return squaredSum**0.5


# This function calculates the vector lengths for all documents
def createVectorTable(TfIdf):
    table = []
    # Copy first row for header labels
    table.append(TfIdf[0])
    # Remove term label
    table[0][0] = "file"
    # Loop through all docuemts ids
    lengthRow = ["length"]
    for documentsId in range(1, len(TfIdf[0])):
        lengthRow.append(calculateVectorLength(TfIdf, documentsId))
    table.append(lengthRow)
    return table


# This function returns the dotproduct of two documents
def createDotProduct(TfIdf, document1, document2):
    dotProductSum = 0
    for line in range(1, len(TfIdf)):
        dotProductSum += TfIdf[line][document1] * TfIdf[line][document2]
    return dotProductSum


# This function returns the simalarity between a number of documents
def compareDocuments(TfIdf, vectorLengthTable, document1, document2):
    dotProduct = createDotProduct(TfIdf, document1, document2)
    lengthProduct = vectorLengthTable[1][document1] * vectorLengthTable[1][document2]
    return dotProduct/lengthProduct


# This function returns the final output table with the ranked results
def createRankedTable(queryTable):
    rankedTable = []
    # Append row with headers
    rankedTable.append(["position", "document", "similarity", "scale", "preview"])
    # Rotate the table so we can sort it
    rotatedTable = []
    for documentId in range(0, len(queryTable[0])-1):
        rotatedTable.append([queryTable[0][documentId], queryTable[1][documentId]])
    # Sort the table
    rankedQueryTable = sorted(rotatedTable, key=lambda score: score[1].real, reverse=True)
    maxResult = limit
    if limit == 0:
        maxResult = len(rankedQueryTable)
    for rank in range(0, maxResult):
        # Find the scale (LOW, MEDIUM, HIGH)
        scale = "medium"
        if rankedQueryTable[rank][1] < 0.2:
            scale = "low"
        if rankedQueryTable[rank][1] > 0.8:
            scale = "high"
        # Append row to table
        if not hideLow:
            rankedTable.append([rank+1, rankedQueryTable[rank][0], rankedQueryTable[rank][1], scale])
        else:
            if rankedQueryTable[rank][1] > 0.01:
                rankedTable.append([rank+1, rankedQueryTable[rank][0], rankedQueryTable[rank][1], scale])
    return rankedTable    


# This function returns a query table
def createQueryTable(TfIdf, VectorLengthTable, queryId):
    table = []
    table.append(TfIdf[0][1:])
    results = []
    for id in range(1, len(TfIdf[0])):
        if not id == queryId:
            results.append(compareDocuments(TfIdf, VectorLengthTable, queryId, id))
    table.append(results)
    return table


# This function finds the text around a word
def textAroundWord(text, word, rangeAround=7):
    if(word == ''):
        return "No preview available..."
    output = "... "
    # Replace newlines
    text = text.replace("\n", " ")
    # Split at spaces
    textArray = text.split(" ")
    # Remove all empty objects
    textArray = list(filter(lambda a: a != "", textArray))
    # Create a all lowercase and lemitised list to find index of word
    processedArray = [lemmatizer.lemmatize(re.sub(r"[“”!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n'0-9]", r"", word).lower()) for word in textArray]
    # Find the first index of the keyword
    wordIndex = processedArray.index(word)
    # Loop through the range around the word
    for x in range(-rangeAround, rangeAround):
        if not x+wordIndex < 0 and not x+wordIndex >= len(textArray):
            if not x == 0:
                output += f"{textArray[wordIndex + x]} "
            else:
                output += f"<b>{textArray[wordIndex]}</b> "
    output += "..."
    return output


# This function adds a preview for every row in the output
def addPreviews(freqTable, rankedTable):
    # Create output table
    output = []
    header = rankedTable[0]
    header.append("preview")
    output.append(header)
    # Loop through every document in ranked table
    for file in rankedTable[1:]:
        preview = "No preview available :("
        fileIndex = 0
        queryIndex = len(freqTable[0]) - 1
        # Loop through name row of frequency table to find correct column
        for documentId in range(1, len(freqTable[0])):
            if file[1] == freqTable[0][documentId]:
                fileIndex = documentId
        # Loop through the frequency table to find the most common word
        wordCount = 0
        word = ""
        for row in freqTable[1:]:
            # Check if the document have common terms
            if row[queryIndex] > 0 and row[fileIndex] > wordCount:
                word = row[0]
                wordCount = row[queryIndex]
        # Find the text around word
        preview = textAroundWord(readText(open(os.path.abspath('saved/' + file[1]))), word)
        outputRow = file
        outputRow.append(preview)
        output.append(outputRow)

    return output


class FrequencyTable():
    mFrequencyTable = []
    mFileDicts = []

    def __init__(self, fileLocations=[], table=None, dicts=None):
        # Check if table is provided
        if not table == None and not dicts == None:
            self.mFrequencyTable = table
            self.mFileDicts = dicts
            return
        # Create a name row that contains all the names
        nameRow = ["term"]
        # Create a list that stores all file dicts
        # Loop through all the files
        for fileLocation in fileLocations:
            # Open the file
            try:
                # Open the file
                file = open(fileLocation)
                printLog(f"Indexing {file.name}")
                text = readText(file)
                # Add the filename to the name row
                nameRow.append(os.path.basename(file.name))
                # Convert to lowercase
                text = text.lower()
                # Remove newline
                text = text.replace("\n", " ")
                # Remove chars
                text = re.sub(r"[“”‘’—!\"#$%&()*+-./:;<=>?@[\]^_`{|}~'0-9]", r"", text)
                # Create a dictinary to store words
                fileWords = dict()
                # Loop through all the worlds
                for word in text.split(" "):
                    # Lemitate the word
                    eWord = lemmatizer.lemmatize(word)
                    # Check if word is stopword or empty
                    if eWord not in stopwords.words("english") and not word == "":
                        # Check if the word already exists in the dict
                        if eWord not in fileWords:
                            fileWords[eWord] = 1
                        else:
                            fileWords[eWord] += 1
                # Append dict to file dicts
                self.mFileDicts.append(fileWords)
                # Close the file
                file.close()
            except Exception as e:
                # File could not be opened
                printLog(f"{fileLocation} could not be opened!\n{e}")
        # Add name row to frequency table
        self.mFrequencyTable.append(nameRow)
        # Merge wordcounts into one dict
        allWords = dict()
        for fileWords in self.mFileDicts:
            allWords.update(fileWords)
        # Loop through all the merged words to create frequency table
        for term in allWords.keys():
            # Create a row and add the term to it in column 0
            termRow = [term]
            # Add the freq to every document
            for documentId in range(0,len(self.mFileDicts)):
                try:
                    termRow.append(self.mFileDicts[documentId][term])
                except:
                    termRow.append(0)
            # Add the term row to the table
            self.mFrequencyTable.append(termRow)
        printLog("New frequency table created:", self.mFrequencyTable, table=True)

    # This function returns a new frequency table object, now including one extra file (no recalculation of other files)
    def createNewTableWith(self, fileLocation):
        # Open the file
        file = open(fileLocation)
        text = readText(file)
        # Create the output table
        outputTable = []
        # Add the name to the name row
        outputTable.append(list(self.mFrequencyTable[0]))
        outputTable[0].append(os.path.basename(file.name))
        # Convert to lowercase
        text = text.lower()
        # Remove newline
        text = text.replace("\n", " ")
        # Remove chars
        text = re.sub(r"[“”‘’—!\"#$%&()*+-./:;<=>?@[\]^_`{|}~'0-9]", r"", text)
        # Loop through all the words of the query file
        fileWords = dict()
        for word in text.split(" "):
            # Lemitate the word
            eWord = lemmatizer.lemmatize(word)
            # Check if it is a stop word and not empty
            if eWord not in stopwords.words("english") and not word == "":
                if eWord not in fileWords:
                    fileWords[eWord] = 1
                else:
                    fileWords[eWord] += 1
        # Add query file dict to the other file dicts
        allDicts = self.mFileDicts.copy()
        allDicts.append(fileWords)
        # Create new all files dict
        allWords = dict()
        for fileWords in allDicts:
            allWords.update(fileWords)
        # Loop through all the merged words to create frequency table
        for term in allWords.keys():
            # Create a row and add the term to it in column 0
            termRow = [term]
            # Add the freq to every document
            for documentId in range(0,len(allDicts)):
                try:
                    termRow.append(allDicts[documentId][term])
                except:
                    termRow.append(0)
            # Add the term row to the table
            outputTable.append(termRow)
        printLog("Frequency table created with file", outputTable, table=True)
        return FrequencyTable(table=outputTable, dicts=allDicts)

    # This function returns a dict holding a pre idf value, the idf value if there was another document, per term
    def createPreIdfDict(self):
        output = dict()
        # Get the number of documents + 1
        N = len(self.mFileDicts) + 1
        # Loop through all the rows
        for termRow in self.mFrequencyTable[1:]:
            df = 0
            for frequency in termRow[1:]:
                if frequency > 0:
                    df += 1
            output[termRow[0]] = log2(N/df)
        return output

    # This function can be called on a frequency matrix that has a query document added to it
    # together with the preIdfTable created with a table that has NO query document in it
    # the last document in the table will be used as query document. It will return a TfIdf table
    def getTfIdf(self, preIdf):
        output = []
        # Get number of documents + 1
        N = len(self.mFrequencyTable)
        # Add name row to output table
        output.append(self.mFrequencyTable[0])
        # Loop through the terms
        for termRow in self.mFrequencyTable[1:]:
            outputRow = [termRow[0]]
            # Check if the tfidf needs to be recalculated
            if termRow[-1] > 0:
                # Recalculate value
                df = 0;
                for freq in termRow[1:]:
                    if freq > 0:
                        df += 1
                idf = log2(N/df)
                for freq in termRow[1:]:
                    outputRow.append(freq * idf)
            else:
                # Recalculation not needed, use preIdf value
                for freq in termRow[1:]:
                    outputRow.append(freq * preIdf[termRow[0]])
            output.append(outputRow)
        printLog("TfIdf table created:", output, table=True)
        return output

    def getFileLen(self):
        return len(self.mFileDicts)

    def getList(self):
        return self.mFrequencyTable


# Main functions

# Create flask app
app = flask.Flask(__name__)

# Landing page
@app.route("/")
def main():
    return render_template('main.html', ip=request.access_route[-1])


# Search API
def allowed_file(filename):
	return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/search", methods=["POST"])
def search():
    printLog("Request started!")
    # Save time for time comaprison
    startTime = time.time()
    # Retreive file
    mFile = request.files['file']
    if not allowed_file(mFile.filename):
        return(json.dumps({'satus': 400, 'error': 'File not supported'}), 400)
    # Save file in tmp directory
    mFilename = str(randint(999999999, 9999999990)) + '.' + mFile.filename.split('.')[-1]
    printLog("File uploaded, save in temp folder as", mFilename)
    mFile.save("tmpUploads/" + mFilename)

    # Create frequency table with query file
    printLog("Creating frequency table with file...")
    queryFreqTable = frequencyTable.createNewTableWith("tmpUploads/" + mFilename)
    printLog("Creating tfIdf table with file...")
    tfIdf = queryFreqTable.getTfIdf(preIdf)

    # Create vector length table
    printLog("Calculating vector lengths...")
    vectorLengths = createVectorTable(tfIdf)
    printLog("Vector length table:", vectorLengths, table=True)

    # Create query table
    printLog("Generating query table")
    queryTable = createQueryTable(tfIdf, vectorLengths, queryFreqTable.getFileLen())
    printLog("Query length table:", queryTable, table=True)

    # Generating output
    printLog("Generating previews and output...")
    mRanked = createRankedTable(queryTable)
    mRanked = addPreviews(queryFreqTable.getList(), mRanked)
    printLog("Output:", mRanked, table=True)

    # Remove tmp file
    os.remove("tmpUploads/" + mFilename)

    printLog(f"Query completed in: {time.time() - startTime} sec")

    # Print memory usage if profiling is enabled
    if memoryProfiling:
        print(h.heap())

    return json.dumps(mRanked)


# This function indexes the documents saved in the folder
def index():
    global frequencyTable
    global preIdf
    printLog("\nIndexing started!")
    # Get all documents in saved folder
    savedFileLocations = os.listdir('saved')
    for fileLocationIndex in range(len(savedFileLocations)):
        savedFileLocations[fileLocationIndex] = "saved/" + savedFileLocations[fileLocationIndex]
    # Create frequency table
    frequencyTable = FrequencyTable(savedFileLocations)
    # Create PreIdfDict
    preIdf = frequencyTable.createPreIdfDict()
    printLog("Indexing complete!")

# Make sure that indexing happens at start
index()

# Start server
if __name__ == "__main__":
    printLog(f"\nStarting sever on port {port}")
    serve(app, host='0.0.0.0', port=port, threads=amountOfThreads)

