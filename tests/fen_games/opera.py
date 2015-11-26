GAME='''
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

r n b q k b n r
p p p p . p p p
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

r n b q k b n r
p p p p . p p p
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . N . .
P P P P . P P P
R N B Q K B . R

r n b q k b n r
p p p . . p p p
. . . p . . . .
. . . . p . . .
. . . . P . . .
. . . . . N . .
P P P P . P P P
R N B Q K B . R

r n b q k b n r
p p p . . p p p
. . . p . . . .
. . . . p . . .
. . . P P . . .
. . . . . N . .
P P P . . P P P
R N B Q K B . R

r n . q k b n r
p p p . . p p p
. . . p . . . .
. . . . p . . .
. . . P P . b .
. . . . . N . .
P P P . . P P P
R N B Q K B . R

r n . q k b n r
p p p . . p p p
. . . p . . . .
. . . . P . . .
. . . . P . b .
. . . . . N . .
P P P . . P P P
R N B Q K B . R

r n . q k b n r
p p p . . p p p
. . . p . . . .
. . . . P . . .
. . . . P . . .
. . . . . b . .
P P P . . P P P
R N B Q K B . R

r n . q k b n r
p p p . . p p p
. . . p . . . .
. . . . P . . .
. . . . P . . .
. . . . . Q . .
P P P . . P P P
R N B . K B . R

r n . q k b n r
p p p . . p p p
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . Q . .
P P P . . P P P
R N B . K B . R

r n . q k b n r
p p p . . p p p
. . . . . . . .
. . . . p . . .
. . B . P . . .
. . . . . Q . .
P P P . . P P P
R N B . K . . R

r n . q k b . r
p p p . . p p p
. . . . . n . .
. . . . p . . .
. . B . P . . .
. . . . . Q . .
P P P . . P P P
R N B . K . . R

r n . q k b . r
p p p . . p p p
. . . . . n . .
. . . . p . . .
. . B . P . . .
. Q . . . . . .
P P P . . P P P
R N B . K . . R

r n . . k b . r
p p p . q p p p
. . . . . n . .
. . . . p . . .
. . B . P . . .
. Q . . . . . .
P P P . . P P P
R N B . K . . R

r n . . k b . r
p p p . q p p p
. . . . . n . .
. . . . p . . .
. . B . P . . .
. Q N . . . . .
P P P . . P P P
R . B . K . . R

r n . . k b . r
p p . . q p p p
. . p . . n . .
. . . . p . . .
. . B . P . . .
. Q N . . . . .
P P P . . P P P
R . B . K . . R

r n . . k b . r
p p . . q p p p
. . p . . n . .
. . . . p . B .
. . B . P . . .
. Q N . . . . .
P P P . . P P P
R . . . K . . R

r n . . k b . r
p . . . q p p p
. . p . . n . .
. p . . p . B .
. . B . P . . .
. Q N . . . . .
P P P . . P P P
R . . . K . . R

r n . . k b . r
p . . . q p p p
. . p . . n . .
. N . . p . B .
. . B . P . . .
. Q . . . . . .
P P P . . P P P
R . . . K . . R

r n . . k b . r
p . . . q p p p
. . . . . n . .
. p . . p . B .
. . B . P . . .
. Q . . . . . .
P P P . . P P P
R . . . K . . R

r n . . k b . r
p . . . q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
R . . . K . . R

r . . . k b . r
p . . n q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
R . . . K . . R

r . . . k b . r
p . . n q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . R

. . . r k b . r
p . . n q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . R

. . . r k b . r
p . . R q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K . . . . R

. . . . k b . r
p . . r q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K . . . . R

. . . . k b . r
p . . r q p p p
. . . . . n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . .

. . . . k b . r
p . . r . p p p
. . . . q n . .
. B . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . .

. . . . k b . r
p . . B . p p p
. . . . q n . .
. . . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . .

. . . . k b . r
p . . n . p p p
. . . . q . . .
. . . . p . B .
. . . . P . . .
. Q . . . . . .
P P P . . P P P
. . K R . . . .

. Q . . k b . r
p . . n . p p p
. . . . q . . .
. . . . p . B .
. . . . P . . .
. . . . . . . .
P P P . . P P P
. . K R . . . .

. n . . k b . r
p . . . . p p p
. . . . q . . .
. . . . p . B .
. . . . P . . .
. . . . . . . .
P P P . . P P P
. . K R . . . .

. n . R k b . r
p . . . . p p p
. . . . q . . .
. . . . p . B .
. . . . P . . .
. . . . . . . .
P P P . . P P P
. . K . . . . .
'''
