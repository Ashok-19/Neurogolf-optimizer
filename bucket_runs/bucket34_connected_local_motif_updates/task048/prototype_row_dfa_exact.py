#!/usr/bin/env python3
from collections import defaultdict
from itertools import combinations
from functools import lru_cache

W=8

def overlaps(a,b):
    r1,c1=a;r2,c2=b
    return not (r1+3<=r2 or r2+3<=r1 or c1+3<=c2 or c2+3<=c1)

seqs=set(); places=[(r,c) for r in range(7) for c in range(7)]
for a,b in combinations(places,2):
    if overlaps(a,b): continue
    rows=[0]*8
    for r,c in (a,b): rows[r]|=3<<c;rows[r+1]|=3<<c
    seqs.add(tuple(rows))
print('seqs',len(seqs),flush=True)

children=[];children.append({})
for s in sorted(seqs):
    n=0
    for x in s:
        if x not in children[n]:
            children[n][x]=len(children);children.append({})
        n=children[n][x]
print('trie nodes',len(children),flush=True)

start_conn=((0,)*W,(),0)

@lru_cache(None)
def trans_conn(state,occ,red):
    labels,flags,closed=state
    nprev=max(labels,default=0)
    p=list(range(nprev+W))
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
    touched=0
    for c in range(W):
        if ((occ>>c)&1) and labels[c]: touched|=1<<(labels[c]-1)
    for i in range(nprev):
        if not ((touched>>i)&1) and flags[i]: closed=min(2,closed+1)
    root_red={}
    for i in range(nprev):
        if flags[i]: root_red[find(i)]=True
    for c in range(W):
        if (red>>c)&1: root_red[find(nprev+c)]=True
    mp={}; out=[]; newflags=[]
    for c in range(W):
        if not (occ>>c)&1: out.append(0);continue
        root=find(nprev+c)
        lab=mp.get(root)
        if lab is None:
            lab=len(mp)+1;mp[root]=lab;newflags.append(bool(root_red.get(root,False)))
        out.append(lab)
    return (tuple(out),tuple(newflags),closed)

def accept(conn):
    labels,flags,closed=conn
    return closed+sum(flags)==1

cur={(start_conn,0):0}
step_states=[cur]
transitions=[]
for row in range(8):
    nxt={}; tr={}
    for (conn,node),sid in cur.items():
        for red,nextnode in children[node].items():
            free=(~red)&255
            sub=free
            while True:
                occ=red|sub
                ns=(trans_conn(conn,occ,red),nextnode)
                nid=nxt.get(ns)
                if nid is None:
                    nid=len(nxt);nxt[ns]=nid
                tr[(sid,(occ<<8)|red)]=nid
                if sub==0:break
                sub=(sub-1)&free
    transitions.append(tr);step_states.append(nxt)
    print('row',row,'cur',len(cur),'trans',len(tr),'nxt',len(nxt),'accept',sum(accept(c) for c,n in nxt),flush=True)
    cur=nxt
print('total transition entries',sum(map(len,transitions)),flush=True)
print('memo',trans_conn.cache_info(),flush=True)
