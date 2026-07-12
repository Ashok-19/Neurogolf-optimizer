#!/usr/bin/env python3
from collections import defaultdict
from itertools import combinations

W=8

def overlaps(a,b,spacing=1):
    (r1,c1),(r2,c2)=a,b
    if r1+2+spacing <= r2: return False
    if r2+2+spacing <= r1: return False
    if c1+2+spacing <= c2: return False
    if c2+2+spacing <= c1: return False
    return True

seqs=set()
places=[(r,c) for r in range(7) for c in range(7)]
for a,b in combinations(places,2):
    if overlaps(a,b,1): continue
    rows=[0]*8
    for r,c in (a,b):
        rows[r]|=3<<c; rows[r+1]|=3<<c
    seqs.add(tuple(rows))
print('red sequences',len(seqs))
allowed=[set() for _ in range(8)]
for s in seqs:
    for i,x in enumerate(s): allowed[i].add(x)
print('allowed red masks', [len(x) for x in allowed])

start=((0,)*W,(),0)

def trans(state,occ,red):
    labels,flags,closed=state
    nprev=max(labels,default=0)
    n=nprev+W
    p=list(range(n))
    def find(x):
        while p[x]!=x:
            p[x]=p[p[x]];x=p[x]
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:p[b]=a
    for c in range(W):
        if not (occ>>c)&1: continue
        nn=nprev+c
        if c and (occ>>(c-1))&1: union(nn,nprev+c-1)
        if labels[c]: union(nn,labels[c]-1)
    touched=[False]*nprev
    for c in range(W):
        if ((occ>>c)&1) and labels[c]: touched[labels[c]-1]=True
    for i in range(nprev):
        if not touched[i] and flags[i]: closed=min(2,closed+1)
    rflag=defaultdict(bool)
    for i in range(nprev):
        if flags[i]: rflag[find(i)]=True
    for c in range(W):
        if (occ>>c)&1 and (red>>c)&1: rflag[find(nprev+c)]=True
    root_to_lab={}; newflags=[]; out=[]
    for c in range(W):
        if not (occ>>c)&1:
            out.append(0);continue
        root=find(nprev+c)
        if root not in root_to_lab:
            root_to_lab[root]=len(root_to_lab)+1
            newflags.append(bool(rflag[root]))
        out.append(root_to_lab[root])
    return (tuple(out),tuple(newflags),closed)

def accept(state):
    labels,flags,closed=state
    return closed+sum(flags)==1

states={start:0}
cur={start}
all_trans=[]
for row in range(8):
    nxt=set(); nt=0
    for st in cur:
        for red in allowed[row]:
            free=(~red)&255
            sub=free
            while True:
                occ=red|sub
                ns=trans(st,occ,red)
                nxt.add(ns);nt+=1
                if sub==0: break
                sub=(sub-1)&free
    for s in nxt:
        if s not in states: states[s]=len(states)
    all_trans.append(nt)
    print('row',row,'cur',len(cur),'nxt',len(nxt),'raw transitions',nt,'global states',len(states), 'accept',sum(map(accept,nxt)))
    cur=nxt
print('total states',len(states),'raw transitions',sum(all_trans))
