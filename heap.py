class Heap(object):

    def __init__(self):
        self.pq = [None]

    def size(self):
        return len(self.pq) - 1

    def swap(self, i, j):
        if (i != j 
                and i <= self.size()
                and j <= self.size()
                and i >= 0 
                and j >= 0):
            self.pq[i], self.pq[j] = self.pq[j], self.pq[i]

    def larger(self, i, j):
        return (self.pq[i][0](self.pq[i][1]) 
            > self.pq[j][0](self.pq[j][1]))

    def swim(self, i):
        while (i > 1 
                and self.larger(i // 2, i)):
            self.swap(i, i // 2)
            i = i // 2

    def sink(self, i):
        while 2 * i <= self.size():
            j = 2 * i
            if (j < self.size()
                    and self.larger(j, j + 1)):
                j += 1
            if not self.larger(i, j):
                break
            self.swap(i, j)
            i = j  

    def put(self, key, value):
        self.pq += [(key, value)]
        self.swim(self.size())

    def peek(self):
        if (self.size() > 0):
            return self.pq[1]
    
    def pop(self):
        if (self.size() > 0):
            x = self.pq[1]
            self.swap(1, self.size())
            self.pq = self.pq[:-1]
            self.sink(1)
            return x
