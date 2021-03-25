import csv
import random

class KnowledgeBase:

    def __init__(self):
        """
            Initialization of the KnowledgeBase class, with the following attributes:

            self.characters - A list of character dictionaries
            self.characteristics - The master list of characteristics and values for all characters
                                    This dictionary should be used to track characteristics to help
                                    in solving the problem.
            self.the_character - The random character to interrogate
        """
        self.characters = ['ALEX', 
                           'ALFRED', 
                           'ANITA', 
                           'ANNE', 
                           'BERNARD', 
                           'BILL', 
                           'CHARLES', 
                           'CLAIRE', 
                           'DAVID', 
                           'ERIC', 
                           'FRANS', 
                           'GEORGE', 
                           'HERMAN', 
                           'JOE', 
                           'MARIA', 
                           'MAX', 
                           'PAUL', 
                           'PETER', 
                           'PHILIP', 
                           'RICHARD', 
                           'ROBERT', 
                           'SAM', 
                           'SUSAN', 
                           'TOM', ]            # A list of all characters, initially (which are dictionaries)
        self.characteristics = {
            'IS_MALE': ['ALEX', 'ALFRED', 'BERNARD', 'BILL', 'CHARLES', 'DAVID', 'ERIC', 'FRANS', 'GEORGE', 'HERMAN', 'JOE', 'MAX', 'PAUL', 'PETER', 'PHILIP', 'RICHARD', 'ROBERT', 'SAM', 'TOM',],
            'IS_FEMALE': ['ANITA', 'ANNE', 'CLAIRE', 'MARIA', 'SUSAN'],
            'BLACK_HAIR': ['ALEX', 'ANNE', 'MAX', 'PHILIP', 'TOM'],
            'RED_HAIR': ['ALFRED', 'BILL', 'CLAIRE', 'FRANS', 'HERMAN', 'ROBERT'],
            'WHITE_HAIR': ['ANITA', 'GEORGE', 'PAUL', 'PETER', 'SAM'],
            'BROWN_HAIR': ['BERNARD', 'MARIA', 'RICHARD'],
            'BLONDE_HAIR': ['CHARLES', 'DAVID', 'ERIC', 'JOE', 'SUSAN'],
            'BALD': ['BILL', 'HERMAN', 'RICHARD', 'SAM', 'TOM'],
            'HAT': ['BERNARD', 'CLAIRE', 'ERIC', 'GEORGE', 'MARIA'],
            'BLUE_EYES': ['ALFRED', 'ANITA', 'PETER', 'ROBERT', 'TOM'],
            'BROWN_EYES': ['ALEX', 'ANNE', 'BERNARD', 'BILL', 'CHARLES', 'CLAIRE', 'DAVID', 'ERIC', 'FRANS', 'GEORGE', 'HERMAN', 'JOE', 'MARIA', 'MAX', 'PAUL', 'PHILIP', 'RICHARD', 'SAM', 'SUSAN', ],
            'MUSTACHE': ['ALEX', 'ALFRED', 'CHARLES', 'MAX', 'RICHARD'],
            'BEARD': ['BILL', 'DAVID', 'PHILIP', 'RICHARD'],
            'GLASSES': ['CLAIRE', 'JOE', 'PAUL', 'SAM', 'TOM'],
            'EARINGS': ['ANNE', 'MARIA', 'SUSAN']
        }       # A dictionary of all known characteristics

        ##-- Read the characters and characteristics
        #self.read_characters()
        self.guess_me = self.characters[random.randint(0, len(self.characters)-1)]


    def read_characters(self):
        """
            Sets up the Character Dictionaries and adds them to the characters list.
            Reads from CSV file characters.csv
        """
        self.characters = []
        with open('characters.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for character in reader:
                self.characters.append(character)


    def tell(self,key,value):
        """
            Tell the KnowledgeBase a new piece of information.
        """
        # Guess me character has the characteristic, so purge all characters left that do not
        if(value):
            for character in self.characters:
                if character not in self.characteristics[key]:
                    self.characters.remove(character)
        # Guess me character does not have the characteristic, so purge all charecters left that do, and drop the trait
        else:
            for character in self.characteristics[key]:
                if character in self.characters:
                    self.characters.remove(character)
            del self.characteristics[key]


    def ask(self,key,value):
        """
            Queries the_character about a specific key, value pair
        """
        if(value):
            if self.guess_me in self.characteristics[key]:
                return True
        else:
            if self.guess_me not in self.characteristics[key]:
                return True

        return False

    def ask_vars(self,key,value):
        """
            Returns the list of remaining characters that meet the key,value pair
        """
        return self.characters
