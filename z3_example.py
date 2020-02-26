from z3 import *

s = Solver()
# # CNF

wl_0_0 = Real('wl_0_0')
wl_0_1 = Real('wl_0_1')
wl_0_2 = Real('wl_0_2')
wl_0_3 = Real('wl_0_3')
wl_0_4 = Real('wl_0_4')
wl_0_5 = Real('wl_0_5')
wl_0_6 = Real('wl_0_6')
wl_0_7 = Real('wl_0_7')
wl_0_8 = Real('wl_0_8')
wl_0_9 = Real('wl_0_9')

wl_1_0 = Real('wl_1_0')
wl_1_1 = Real('wl_1_1')
wl_1_2 = Real('wl_1_2')
wl_1_3 = Real('wl_1_3')
wl_1_4 = Real('wl_1_4')
wl_1_5 = Real('wl_1_5')
wl_1_6 = Real('wl_1_6')
wl_1_7 = Real('wl_1_7')
wl_1_8 = Real('wl_1_8')
wl_1_9 = Real('wl_1_9')

wl_2_0 = Real('wl_2_0')
wl_2_1 = Real('wl_2_1')
wl_2_2 = Real('wl_2_2')
wl_2_3 = Real('wl_2_3')
wl_2_4 = Real('wl_2_4')
wl_2_5 = Real('wl_2_5')
wl_2_6 = Real('wl_2_6')
wl_2_7 = Real('wl_2_7')
wl_2_8 = Real('wl_2_8')
wl_2_9 = Real('wl_2_9')


M = AstMap() # 字典
r_0=Int('r_0')
M[r_0] = RealVal(0)
r_1=Int('r_1')
M[r_1] = RealVal(0)
r_2=Int('r_2')
M[r_2] = RealVal(0)
r_3=Int('r_3')
M[r_3] = RealVal(0)
r_4=Int('r_4')
M[r_4] = RealVal(0)
r_5=Int('r_5')
M[r_5] = RealVal(0)
r_6=Int('r_6')
M[r_6] = RealVal(0)
r_7=Int('r_7')
M[r_7] = RealVal(0)
r_8=Int('r_8')
M[r_8] = RealVal(0)
r_9=Int('r_9')
M[r_9] = RealVal(0)


WL_0 = [Real('wl_%s_0' % i) for i in range(3)]
WL_1 = [Real('wl_%s_1' % i) for i in range(3)]
WL_2 = [Real('wl_%s_2' % i) for i in range(3)]
WL_3 = [Real('wl_%s_3' % i) for i in range(3)]
WL_4 = [Real('wl_%s_4' % i) for i in range(3)]
WL_5 = [Real('wl_%s_5' % i) for i in range(3)]
WL_6 = [Real('wl_%s_6' % i) for i in range(3)]
WL_7 = [Real('wl_%s_7' % i) for i in range(3)]
WL_8 = [Real('wl_%s_8' % i) for i in range(3)]
WL_9 = [Real('wl_%s_9' % i) for i in range(3)]

RL = [Int('r_%s' % i) for i in range(10)]

for i in WL_0:
    M[r_0] += i
for i in WL_1:
    M[r_1] += i
for i in WL_2:
    M[r_2] += i
for i in WL_3:
    M[r_3] += i
for i in WL_4:
    M[r_4] += i
for i in WL_5:
    M[r_5] += i
for i in WL_6:
    M[r_6] += i
for i in WL_7:
    M[r_7] += i
for i in WL_8:
    M[r_8] += i
for i in WL_9:
    M[r_9] += i

index = r_0
out = M[r_0]

#找到字典中value值最大的 index
for i in RL:
    index = If(M[i] > out, i, index)
    out = If(M[i] > out, M[i], out)
s.add(And(r_0==0, r_1==1, r_2==2, r_3==3, r_4==4, r_5==5, r_6==6, r_7==7, r_8==8, r_9==9))
s.add(index != 0)
s.add(And(wl_0_0==99.9, wl_0_1==0, wl_0_2==0, wl_0_3==0, wl_0_4==0, wl_0_5==0, wl_0_6==0, wl_0_7==0, wl_0_8==0, wl_0_9==0))
s.add(And(wl_1_0==0, wl_1_1==1.0, wl_1_2==0, wl_1_3==0, wl_1_4==0, wl_1_5==0, wl_1_6==0, wl_1_7==0, wl_1_8==0, wl_1_9==0))
s.add(And(wl_2_0==0, wl_2_1==0, wl_2_2==0, wl_2_3==0, wl_2_4==0, wl_2_5==0, wl_2_6==0, wl_2_7==0, wl_2_8==0, wl_2_9==0))

if unsat == s.check():
    print('unsat')
else:
    print('sat')
    print(s.model())
