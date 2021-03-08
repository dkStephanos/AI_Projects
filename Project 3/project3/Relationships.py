import json
from KnowledgeBase import KnowledgeBase

class Relationships:

    def __init__(self):
        self.kb = KnowledgeBase()
        with open("relationships.json") as f:
            d = json.loads(f.read())
            self.parents = d['parent']


    def parent(self,x,y):
        return {x:y} in self.parents


    def father(self,x,y):
        raise NotImplementedError()


    def mother(self,x,y):
        raise NotImplementedError()
