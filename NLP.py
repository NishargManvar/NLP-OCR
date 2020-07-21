#import needed files
import csv;
import nltk;
import math;
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import WhitespaceTokenizer 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# initalising certain variables
tk = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
tkw =WhitespaceTokenizer() ;
stemmer = PorterStemmer();

count=0;	#variable to ignore the first line of the csv file
maindict = {};	#dictionary with all unique words
wordnumber = 0;	#variable to give numbers to words in dictionary
success = 0;	#variable to count success finaly
failure = 0;	#variable to count failures

#LIST OF FUNCTIONS
#****************************************************************************************************************************
#call function for the first step of code...make tokens and update dictionary
def feedinput(sentence):
	tokens = maketoken(sentence);	#call to make tokens
	givenumber(tokens);		#update tokens into the dictionary

#function returns tokens given a array
def maketoken(sentence):
	tokens=tk.tokenize(sentence);	#use tokenize functions to tokenize the sentence-array
	return tokens;

#function to update given word tokens in the dictionary
def givenumber(tokens):
	global wordnumber;
	for word in tokens:
		if stemming(lower(word)) in maindict:	#check if word already exists in the dictionary
			continue;		#if yes then continue
		maindict[stemming(lower(word))]=wordnumber;	#if no add word and give it's value in the dictionary
		wordnumber = wordnumber + 1;

#function for stemming the string given as input 
def stemming(string):
	return stemmer.stem(string);	#stem or lemmatize the given string using functions

#function to make the given string lower code
def lower(string):
	return string.lower();	#lowercase the string using inbuilt function

#function to make bag of words for corresponding input tokens
def getbagofwords(tokens):
	global bagofwords;
	bagofwords = [0]*wordnumber;	#initalize the array for each time for each input
	for word in tokens:
		value = maindict[stemming(lower(word))];	
		bagofwords[value] = bagofwords[value]+1;	#increase the count in the bagofwords corresponding to the words

#call function to forward propogate and back propogate 'iterations' number of times
def propogate(expected,iterations):
	for i in range(iterations):
		y = output();	
		finaloutput = sigmoid(y); #get output
		err = error(finaloutput,expected);	#calculate error
		updateweights(err,y,finaloutput,expected);	#update weights using back propogation

#function to calculate the optput by summing the product of count and their corresponding weights
def output():
	global bagofwords,weights;
	sumvalue = 0;
	for j in range(len(weights)):
		sumvalue = sumvalue + (bagofwords[j]*weights[j]);	#calculate the optput by summing the product of count and their corresponding weights 
	return sumvalue;

#function to return sigmoid value of given input
def sigmoid(sum):
	return 1/(1+math.exp(-1*sum));

#function to calculate the error from output and expected value
def error(output,expected):
	return 0.5*((output - expected)**2);	#error expression

#function to backpropogate and update weights
def updateweights(error,sumvalue,finaloutput,expected):
	global weights,bagofwords;
	for i in range(len(weights)):
		changeinweight = -1*(expected-finaloutput)*(math.exp(-1*sumvalue))*bagofwords[i]/((1+(math.exp(-1*sumvalue)))**2);	#change in weights expression
		weights[i]= weights[i] - (0.05*changeinweight);	#update weight (can change the LR)
#********************************************************************************************************************

#open csv file first time
csvfile = open('imdb_small.csv');
csvreader = csv.reader(csvfile);

#loop to make the tokens and update dictionary
for row in csvreader:
	if count==0:
		count = count + 1;
		continue;	#steps to ignore first line of csv file
	feedinput(row[0]);

#global variables
global weights ;	#array of weights
weights = [0]*wordnumber;	#initalising the weights
global bagofwords;	#array of bagofwords for corresponding input

#opening the csv file again
csvfile1 = open('imdb_small.csv');
csvreader1 = csv.reader(csvfile1);
count = 0 ;
#loop to input each comment into the neural network and update the weights
for row1 in csvreader1:
	if count==0:
		count = count + 1;
		continue;	#steps to ignore first line of csv file
	getbagofwords(maketoken(row1[0]));
	if row1[1]=="negative":
		propogate(0,1);
	else:
		propogate(1,1);

#opening the csv file for third time
csvfile2 = open('imdb_small.csv');
csvreader2 = csv.reader(csvfile2);
count = 0;
#loop to reinput the comements and check the accuracy
for row2 in csvreader2:
	if count==0:
		count = count + 1;
		continue;	#steps to ignore first line of csv file
	getbagofwords(maketoken(row2[0]));
	y = output();
	finaloutput = sigmoid(y);
	if finaloutput>0.5 and row2[1]=="positive":
		success = success + 1;
	elif finaloutput<0.5 and row2[1]=="negative":
		success = success + 1;
	elif finaloutput == 0.5:
		print("What a co-incidence");
	else:
		failure = failure + 1;

#print final results
print("success :",success);
print("failures :",failure);
persuccess = success/(success+failure);
print("% success = ",persuccess*100);