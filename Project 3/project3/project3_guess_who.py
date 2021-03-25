from KnowledgeBase import KnowledgeBase
import random

def main():
    """
        Driver Method
    """
    kb = KnowledgeBase()
    print("Starting game, Guess Me characters is: ", kb.guess_me)

    while(len(kb.characters) > 1):
        value = bool(random.getrandbits(1))
        key = random.choice(list(kb.characteristics.keys()))
        print("Ask: ", key, value)

        answer = kb.ask(key, value)
        # If we were right (i.e. we guessed a correct trait) or we guess false and wrong (i.e. we say X doesn't have A, but they do), then tell the knowledgebase the guess me character has the trait
        if (answer and value) or (not answer and not value):
            kb.tell(key, True)
        else: # Otherwise, tell the knowledgebase the guess me character does not have the trait
            kb.tell(key, False)
        print("Tell: ", key, value)

        print("Characteristics remaining", kb.characteristics)
        print("Characters remaining", kb.characters)

    print('Found character: ', kb.characters[0])
    print('Correct character: ', kb.guess_me)
    if(kb.guess_me == kb.characters[0]): 
        print("YOU WIN!!!!!")
    else:
        print("BLAME THE COMPUTER!!!!")


main()
