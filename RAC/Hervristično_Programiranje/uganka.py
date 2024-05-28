class Stanje:
    def __init__(self, stanje):
        self.stanje = stanje
        self.prazno_mesto = [(i, j) for i, v in enumerate(self.stanje) for j, vr in enumerate(v) if self.stanje[i][j] is None][0]

    def __repr__(self):
        return f"Stanje({self.stanje})"
    
    def __str__(self):
        niz = ""
        for vrstica in self.stanje:
            niz += str(vrstica) + "\n"
        return niz.strip()
    
    def __eq__(self, other):
        return self.stanje == other.stanje

    def sosednja_stanja(self):
        i, j = self.prazno_mesto
        smeri = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in smeri:
            ni = i + x
            nj = j + y
            if 0 <= ni < len(self.stanje) and 0 <= nj < len(self.stanje):
                novo_stanje = [v[:] for v in self.stanje]
                novo_stanje[i][j] = self.stanje[ni][nj]
                novo_stanje[ni][nj] = None
                yield (Stanje(novo_stanje), 1)  # To je generator, 1 predstavlja ceno premika

uganka1 = [[4, None, 1], [7, 2, 3], [5, 8, 6]]
uganka2 = [[2, None, 7], [5, 1, 3], [4, 8, 6]]
uganka3 = [[7, 1, 3], [2, 4, None], [5, 8, 6]] 
uganka4 = [[2, 5, 3], [1, 7, 6], [None, 4, 8]]




# 1.
def resitev(n):
    matrika = [[(j + n * i + 1) for j in range(n)] for i in range(n)]
    matrika[-1][-1] = None
    return matrika




class Uganka:
    def __init__(self, matrika):
        self.stanje = Stanje(matrika)
        self.n = len(matrika)

    def zacetno(self):
        return self.stanje
    
    def resitev(self, s):
        return s == Stanje(resitev(self.n))
    
    def razvejaj(self, s):
        for sosed in s.sosednja_stanja():
            yield sosed

    def h(self, s): # Izračuna hevristiko stanja
        r = resitev(self.n)
        odstopanja = 0
        for i in range(self.n):
            for j in range(self.n):
                if s.stanje[i][j] != r[i][j]:
                    odstopanja += 1
        return odstopanja
    
class Uganka2(Uganka):
    def h(self, s):
        r = resitev(self.n)
        odstopanja = 0
        for i in range(self.n):
            for j in range(self.n):
                v = s.stanje[i][j]
                if v is not None:
                    pi, pj = (v - 1) // self.n, (v - 1) % self.n
                else:
                    pi, pj = self.n - 1, self.n - 1
                odstopanja += abs(i - pi) + abs(j - pj)
        return odstopanja


from preiskovanje import Preisci

p = Preisci(Uganka(uganka1), "a*")
p.preisci()
print(p)

p = Preisci(Uganka2(uganka1), "a*")
p.preisci()
print(p)