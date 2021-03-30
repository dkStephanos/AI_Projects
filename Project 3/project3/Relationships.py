import json
from KnowledgeBase import KnowledgeBase

class Relationships:

    def __init__(self):
        self.kb = KnowledgeBase()
        self.kb.read_characters()
        with open("relationships.json") as f:
            d = json.loads(f.read())
            self.parents = d['parent']

    def get_parents(self, x):
        parents = []
        for parent in self.parents:
            if x in list(parent.keys()):
                parents.append(list(parent.values())[0])

        return parents

    def get_children(self, x):
        children = []
        for parent in self.parents:
            if list(parent.values())[0] == x:
                children.append(list(parent.keys())[0])
        
        return children

    def get_siblings(self, x):
        parents = self.get_parents(x)
        if(len(parents) == 0):
            return []
        return self.get_children(parents[0])

    def is_male(self, x):
        return x.upper() in self.kb.characteristics['IS_MALE']

    def is_female(self, x):
        return x.upper() in self.kb.characteristics['IS_FEMALE']

    def parent(self,x,y):
        return {x:y} in self.parents


    def father(self,x,y):
        return self.parent(x,y) and self.is_male(x)


    def mother(self,x,y):
         return self.parent(x,y) and self.is_female(x)

    def grandparent(self,x,y):
        kids = self.get_children(x)
        if len(kids) == 0:
            return False

        for kid in kids:
            grandkids = self.get_children(kid)
            if y in grandkids:
                return True

        return False

    def great_grandparent(self,x,y):
        kids = self.get_children(x)
        if len(kids) == 0:
            return False

        for kid in kids:
            grandkids = self.get_children(kid)
            for grandkid in grandkids:
                great_grandkids = self.get_children(grandkid)
                if len(great_grandkids) == 0:
                    return False
                if y in great_grandkids:
                    return True

        return False

    def grandfather(self,x,y):
        return self.is_male(x) and self.grandparent(x,y)

    def great_grandfather(self,x,y):
        return self.is_male(x) and self.great_grandparent(x,y)

    def grandmother(self,x,y):
        return self.is_female(x) and self.grandparent(x,y)

    def great_grandmother(self,x,y):
        return self.is_female(x) and self.great_grandparent(x,y)

    def spouse(self,x,y):
        kids_x = self.get_children(x)
        kids_y = self.get_children(y)

        return kids_x == kids_y

    def sibling(self,x,y):
        parents_x = self.get_parents(x)
        parents_y = self.get_parents(y)

        return parents_x == parents_y

    def sister(self,x,y):
        parents_x = self.get_parents(x)
        parents_y = self.get_parents(y)

        return parents_x == parents_y and self.is_female(x)

    def brother(self,x,y):
        parents_x = self.get_parents(x)
        parents_y = self.get_parents(y)

        return parents_x == parents_y and self.is_male(x)

    def niece(self,x,y):
        parents = self.get_parents(x)

        is_niece = False
        for parent in parents:
            if y in self.get_siblings(parent):
                is_niece = True

        return is_niece and self.is_female(x)

    def nephew(self,x,y):
        parents = self.get_parents(x)

        is_nephew = False
        for parent in parents:
            if y in self.get_siblings(parent):
                is_nephew = True

        return is_nephew and self.is_male(x)

    def cousin(self,x,y):
        parents = self.get_parents(x)
        print(parents)
        parents_siblings = []
        for parent in parents:
            parents_siblings += (self.get_siblings(parent))
        print(parents_siblings)
        cousins = []
        for sibling in parents_siblings:
            cousins += (self.get_children(sibling))

        is_cousin = False
        print(cousins)
        for cousin in cousins:
            if cousin == y:
                is_cousin = True

        return is_cousin