from KnowledgeBase import KnowledgeBase
import random

def main():
    """
        Driver Method
    """
    play_again = True
    while(play_again == True):
        kb = KnowledgeBase()
        print("Starting game, Guess Me characters is: ", kb.guess_me)
        num_questions = 0
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
            num_questions += 1

        print('Found character: ', kb.characters[0])
        print('Correct character: ', kb.guess_me)
        print('Number of questions asked: ', num_questions)
        if(kb.guess_me == kb.characters[0]): 
            print("YOU WIN!!!!!")
        else:
            print("BLAME THE COMPUTER!!!!")

        print("Play again? [Y] Press any to quit.")
        answer = input()
        if(answer != 'Y' and answer != 'y'):
            play_again = False


main()
