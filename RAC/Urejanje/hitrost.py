def uredi_z_izbiranjem(a):
    for i in range(0, len(a)-1):
        for j in range(i+1, len(a)):
            if a[j] < a[i]:
                a[i], a[j] = a[j], a[i]
        
if __name__ == '__main__':
    import timeit
    import random
    t = list(range(0,1000))
    random.shuffle(t)
    print(timeit.timeit("uredi_z_izbiranjem(t)",
                        number=100,
                        setup="from __main__ import t, uredi_z_izbiranjem"))