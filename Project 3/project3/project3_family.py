from Relationships import Relationships



def main():
    r = Relationships()
    print(r.father("Anita","Paul"))

    print(r.get_children("Alex"))

main()
