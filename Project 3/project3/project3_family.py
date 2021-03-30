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

    print(r.cousin('Philip', 'Tom'))


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
                    print(relationship_dispatch_tb[relationship](personA, personB))
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
