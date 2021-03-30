from Relationships import Relationships
from KnowledgeBase import KnowledgeBase


def main():
    r = Relationships()
    kb = KnowledgeBase()

    relationship_dispatch_tb = {
    'Spouse': r.spouse,
    'Sibling': r.sibling,
    'Sister': r.sister,
    'Brother': r.brother,
    'Niece': r.niece,
    'Nephew': r.nephew,
    'Cousin': r.cousin,
    'Parent': r.parent,
    'Mother': r.mother,
    'Father': r.father,
    'Grandparent': r.grandparent,
    'Grandfather': r.grandfather,
    'Grandmother': r.grandmother,
    'Great-Grandparent': r.great_grandparent,
    'Great-Grandmother': r.great_grandmother,
    'Great-Grandfather': r.great_grandfather,
    }

    '''
    with open("family-tree.csv", 'w') as csvFile:
        for character in kb.characters:
            for relationship in relationship_dispatch_tb.keys():
                for char_to_test in kb.characters:
                    if(relationship_dispatch_tb[relationship](character.lower().capitalize(), char_to_test.lower().capitalize()) == True):
                        print(relationship, character, char_to_test)
                        csvFile.write(f'{relationship}, {character}, {char_to_test}\n')
    '''

    while(True):
        print("Enter the name of the relationship you want to check: ")
        relationship = input()
        if(relationship in relationship_dispatch_tb.keys()):
            while(True):
                print("Enter the name of the first person: ")
                personA = input()
                print("Enter the second of the first person: ")
                personB = input()

                if(personA.upper() in kb.characters and personB.upper() in kb.characters):
                    print(relationship_dispatch_tb[relationship](personA.upper(), personB.upper()))
                else:
                    print("Characters not found in KnowledgeBase base, try again. Here are the possible choices: ", kb.characters)
                print("Press B to go back to relationship selection, E to exit, or any other character to run this relationship again: ")
                choice = input()
                if(choice == 'B'):
                    break
                if(choice == 'E'):
                    exit(0)
        else:
            print("Not a valid selection, try again. Here are the possible choices: ", relationship_dispatch_tb.keys())


main()
