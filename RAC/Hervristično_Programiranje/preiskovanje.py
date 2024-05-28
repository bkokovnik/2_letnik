class Preisci:
    """Razred isci, ki implementira splošni preiskovalni algoritem
    in štiri strategije iskanja:
    (1) iskanje v globino, si = 'v-globino',
    (2) iskanje v širino, si = 'v-širino',
    (3) najprej najboljši, si = 'najprej-najboljsi',
    (4) A*, si = 'a*'."""

    def __init__(self, pr, si):
        """Naredimo novo instanco preiskovalnega algoritma za
        prostor možnih rešitev pr in strategijo iskanja si.
        Predpostavka je, da je pr objekt razreda, ki definira
        naslednje štiri metode:
        (1) zacetno(): vrne začetno stanje v prostoru možnih rešitev;
        (2) resitev(s): preveri ali je stanje s rešitev;
        (3) razvejaj(s): vrne iterator skozi naslednike stanja s;
        (4) h(s): vrne vrednost hevristične funkcije za stanje s."""

        self.pr = pr
        self.si = si

        # Odprti seznam dvojic
        # (1) stanje s, ki še ni razvejano, in
        # (2) cena g dosedanje poti do stanja s
        self.os = [(self.pr.zacetno(), 0)]

        # Zaprti seznam že razvejanih stanj
        self.zs = []

        # Števec razvejanj vozlišč
        self.n_razvejana = 0


    def preisci(self):
        """Splošni preiskovalni algoritem."""

        # Neskončna zanka preiskovanja
        while True:
            #print(self)

            # Vzamemo prvi element iz odprtega seznama os (s je stanje, g je cena poti do s)
            (s, g) = self.os[0]

            # Če je s rešitev vrnemo stanje, ceno poti in število razvejanih vozlišč
            if self.pr.resitev(s):
                return (s, g, self.n_razvejana)
            #else

            # pripravimo se za razvejanje stanja s
            # povečamo število razvejanj in prestavimo s iz odprtega na zaprti seznam
            self.n_razvejana += 1
            self.os = self.os[1:]
            if s not in self.zs: self.zs.append(s)

            # razvejamo stanje s
            for (ns, cena) in self.pr.razvejaj(s):
                # če je naslednje stanje ns na zaprtem seznamu, ga preskočimo
                if ns in self.zs: continue
                # else

                # če gre za iskanje v širino,
                # dodaj novo stanje na konec odprtega seznama
                # sicer pa ga dodaj na začetek
                if self.si == "v-sirino":
                    self.os = self.os + [(ns, g + cena)]
                else:
                    self.os = [(ns, g + cena)] + self.os

            # če gre za strategijo najprej najboljši,
            # uredi odprti seznam po naraščajoči vrednosti g
            if self.si == "najprej-najboljsi":
                self.os = sorted(self.os, key = lambda sg: sg[1])

            # če gre za strategijo a*,
            # uredi odprti seznam po naraščajoči vrednosti g + h
            if self.si == "a*":
                self.os = sorted(self.os, key = lambda sg: sg[1] + self.pr.h(sg[0]))


    def __str__(self):
        """Vrni berljivo predstavitev stanja preiskovalnega algoritma."""

        return f"Zaprti seznam {self.zs}\nOdprti seznam {self.os}\n{self.n_razvejana} razvejanih vozlišč"