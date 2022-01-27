import csv
from math import log2
import os
import sys
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

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Making sure all required nltk packages are installed
print("Making sure all required nltk packages are installed")
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Read arguments from shell
verbose = False
try:
    if sys.argv[1] == "-v":
        verbose = True
except:
    pass


# Simple logging function
def printLog(*input, table=False):
    if verbose and not table:
        print("\n", end="")
        for item in input:
            print(item, end=" ")
        print("\n", end="")
    if verbose and table:
        print("\n", input[0], "\n", end="")
        print(tabulate(input[1], headers="firstrow"))
        print("\n", end="")


def importTfIdf(fileLocation):
    # Open the csv file
    try:
        file = open(fileLocation, 'r')
        csvReader = csv.reader(file)
        # Create a table to save data
        table = []
        for line in csvReader:
            # Change all strings to floats
            row = []
            for item in line:
                try:
                    row.append(float(item))
                except:
                    # Item is not a float, leave as it
                    row.append(item)
            # For every line (term) extract the data per document
            table.append(row)
        return table
    except:
        print("Could not open TfIdf table file...")


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
    for rank in range(0, len(rankedQueryTable)):
        # Find the scale (LOW, MEDIUM, HIGH)
        scale = "medium"
        if rankedQueryTable[rank][1] < 0.1:
            scale = "low"
        if rankedQueryTable[rank][1] > 0.8:
            scale = "high"
        rankedTable.append([rank+1, rankedQueryTable[rank][0], rankedQueryTable[rank][1], scale])

    return rankedTable    


# This function returns the TfIdf (and frequency) table based on the locations of the files
def createTfIdfFromFiles(locations):
    if verbose:
        print("\nCreating tfidf matrix of the following files:", locations)
    # Create the output, idf and frequency table
    output = []
    idf = [["term", "df", "N/df", "idf"]]
    frequency = []
    df = dict()
    # Create a header row
    nameRow = ["term"]
    # A list to store a word dict per file
    fileDicts = []
    # Loop through all the files
    for fileLocation in locations:
        # Open the file
        mFile = open(fileLocation)
        # Add the filename to the name row
        nameRow.append(os.path.basename(mFile.name))
        # Read the text from the file
        mText = mFile.read()
        # Convert to lower case
        mText = mText.lower()
        # Remove chars
        mText = re.sub(r"[“”‘’—!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n'0-9]", r"", mText)
        # Loop through the words
        mWords = dict()
        for word in mText.split(" "):
            # Lemitate the word
            mWord = lemmatizer.lemmatize(word)
            # Check if it is a stop word
            if mWord not in stopwords.words("english") and not word == "":
                # Check if the world already excists in dictionary
                if mWord not in mWords:
                    mWords[mWord] = 1
                else:
                    mWords[mWord] += 1
        # Append wordcount dict to the list of file dicts
        fileDicts.append(mWords)
        # Close the file
        mFile.close()
    # Add the name row to the output and frequency table
    frequency.append(nameRow)
    output.append(nameRow)
    # Merge the wordcounts per document togather in one dict
    mergedDict = dict()
    for fileDict in fileDicts:
        mergedDict.update(fileDict)
    # Loop through all the terms to create the frequency matrix
    for term in mergedDict.keys():
        # Add the term itself to the term column
        termRow = [term]
        # Loop through the documents
        for documentId in range(0, len(fileDicts)):
            try:
                termRow.append(fileDicts[documentId][term])
                # Add to the df dict
                if term in df:
                    df[term] += 1
                else:
                    df[term] = 1
            except:
                termRow.append(0)
        # Add the term row to the frequency table
        frequency.append(termRow)
    if verbose:
        print("\nFrequency table:\n", tabulate(frequency, headers="firstrow"))
    N = len(fileDicts)
    # Loop through all terms again to create the idf table
    for term in mergedDict.keys():
        idf.append([term, df[term], N/df[term], log2(N/df[term])])

    if verbose:
        print("\nIdf table:\n", tabulate(idf, headers="firstrow"))
    # Create output table by multiplying frequency table by idf
    for row in frequency[1:]:
        outputRow = [row[0]]
        for freq in row[1:]:
            outputRow.append(freq*idf[frequency.index(row)][3])
        output.append(outputRow)

    if verbose:
        print("\nOutput table:\n", tabulate(output, headers="firstrow"))
    return [output, frequency]


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
    print(f"word: {word} index: {wordIndex}")
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
        preview = textAroundWord(open('saved/' + file[1]).read(), word)
        outputRow = file
        outputRow.append(preview)
        output.append(outputRow)

    return output


# Create flask app
app = flask.Flask(__name__)


# Landing page
@app.route("/")
def main():
    return render_template('main.html', ip=request.remote_addr)


# Search API
def allowed_file(filename):
	return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/search", methods=["POST"])
def search():
    printLog("Request started!")
    # Retreive file
    mFile = request.files['file']
    if not allowed_file(mFile.filename):
        return(json.dumps({'satus': 400, 'error': 'File not supported'}), 400)
    # Save file in tmp directory
    mFilename = str(randint(999999999, 9999999990)) + '.' + mFile.filename.split('.')[-1]
    printLog("File uploaded, save in temp folder as", mFilename)
    mFile.save("tmpUploads/" + mFilename)

    # Get all documents in saved folder
    savedFileLocations = os.listdir('saved')
    for fileLocationIndex in range(len(savedFileLocations)):
        savedFileLocations[fileLocationIndex] = "saved/" + savedFileLocations[fileLocationIndex]
    # Append own document
    savedFileLocations.append("tmpUploads/" + mFilename)

    # Create TfIdf matrix and frequency table
    mOutput = createTfIdfFromFiles(savedFileLocations)
    mTfIdf = mOutput[0]
    mFrequency = mOutput[1]

    # Create vector length table
    mVectorLengths = createVectorTable(mTfIdf)
    printLog("Vector length table:", mVectorLengths, table=True)

    # Create query table
    mQueryTable = createQueryTable(mTfIdf, mVectorLengths, len(savedFileLocations))
    printLog("Query length table:", mQueryTable, table=True)

    # Create previews
    # Not optimised because every file is checked even with no common words
    # mPreviews = createPreviewDict(savedFileLocations)

    # Create output
    mRanked = createRankedTable(mQueryTable)
    mRanked = addPreviews(mFrequency, mRanked)
    printLog("Output:", mRanked, table=True)


    # Remove tmp file
    os.remove("tmpUploads/" + mFilename)

    return json.dumps(mRanked)


# Start server
if __name__ == "__main__":
    serve(app, host='0.0.0.0', port='8000')
