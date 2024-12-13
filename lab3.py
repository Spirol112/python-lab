import json
from dataclasses import dataclass
from collections import defaultdict, deque

#Mnożenie macierzy
def multiply_matrices(A, B):
    # Liczba kolumn w A musi równać się liczbie wierszy w B
    if len(A[0]) != len(B):
        raise ValueError("Liczba kolumn w macierzy A musi równać się liczbie wierszy w macierzy B.")

    # Inicjalizacja wynikowej macierzy
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):  # Wiersze A
        for j in range(len(B[0])):  # Kolumny B
            for k in range(len(B)):  # Kolumny A i wiersze B
                result[i][j] += A[i][k] * B[k][j]
    return result

#dane osobowe
@dataclass
class Person:
    first_name: str
    last_name: str
    address: str
    postal_code: str
    pesel: str

    def save_to_json(self, filename):
        f = open(filename, 'w')
        json.dump(self.__dict__, f)
        f.close()

    @staticmethod
    def load_from_json(filename):
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        return Person(**data)


# Zadanie 3: Algorytm Dijkstry
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()

    while len(visited) < len(graph):
        min_dist_node = None
        for node in graph:
            if node not in visited:
                if min_dist_node is None or distances[node] < distances[min_dist_node]:
                    min_dist_node = node

        if min_dist_node is None:
            break
        for neighbor, weight in graph[min_dist_node].items():
            if neighbor not in visited:
                new_dist = distances[min_dist_node] + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
        visited.add(min_dist_node)

    return distances
#klasa AhoCorasick
class AhoCorasick:
    def __init__(self):
        self.trie = defaultdict(dict)
        self.output = defaultdict(list)
        self.fail = {}

    def add_word(self, word):
        node = 0
        for char in word:
            node = self.trie[node].setdefault(char, len(self.trie))
        self.output[node].append(word)

    def build(self):
        self.fail = {}
        queue = deque()

        for node in self.trie[0].values():
            self.fail[node] = 0
            queue.append(node)

        while queue:
            current = queue.popleft()
            for char, child in self.trie[current].items():
                queue.append(child)
                fallback = self.fail[current]
                while fallback and char not in self.trie[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.trie[fallback].get(char, 0)
                self.output[child].extend(self.output[self.fail[child]])

    def search(self, text):
        node = 0
        results = []
        for i, char in enumerate(text):
            while node and char not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(char, 0)
            if self.output[node]:
                for match in self.output[node]:
                    results.append((i - len(match) + 1, match))
        return results

#klasa automat
class State:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.transitions = {}

    def add_transition(self, input_symbol, next_state):
        self.transitions[input_symbol] = next_state

    def next_state(self, input_symbol):
        return self.transitions.get(input_symbol, None)

class MooreMachine:
    def __init__(self, states, initial_state):
        self.states = states
        self.current_state = initial_state

    def process(self, inputs):
        outputs = []
        for symbol in inputs:
            outputs.append(self.current_state.output)
            self.current_state = self.current_state.next_state(symbol)
        outputs.append(self.current_state.output)
        return outputs



# Zadanie 6: Dekorator zamieniający litery na duże
def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()

    return wrapper


@uppercase_decorator
def print_message(message):
    return message



# Testowanie zadania 1: Mnożenie macierzy
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
result = multiply_matrices(A, B)
print("Wynik mnożenia macierzy A i B:", result)


# Testowanie zadania 2: Klasa Person
person = Person("grzegorz", "spirytulski", "ul. mila 1", "11-032", "03320907839")
person.save_to_json("person.json")
loaded_person = Person.load_from_json("person.json")
print("Wczytane dane:", loaded_person)

#  Algorytm Dijkstry
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

distances = dijkstra(graph, 'C')
print("Odległości od wierzchołka C:", distances)


# Przykład automaty
s1 = State("S1", "0")
s2 = State("S2", "1")

s1.add_transition("a", s2)
s2.add_transition("b", s1)

moore_machine = MooreMachine([s1, s2], s1)
print(moore_machine.process(["a", "b", "a", "b"]))

# Przykład AhoCorasick
ac = AhoCorasick()
ac.add_word("he")
ac.add_word("she")
ac.add_word("his")
ac.add_word("hers")
ac.build()

print(ac.search("ushers"))


#  Dekorator uppercase
print(print_message("Hello World"))




