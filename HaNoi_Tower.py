def HaNoi_tow(N,from_tow,alt_tow,to_tow):
    if N!=0:
        HaNoi_tow(N-1,from_tow,to_tow,alt_tow)
        print('move disk %d from %d to %d' % (N, from_tow, to_tow) )
        HaNoi_tow(N-1,alt_tow,from_tow,to_tow)
HaNoi_tow(5,1,2,3)