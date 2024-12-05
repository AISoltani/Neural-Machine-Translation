from __future__ import unicode_literals, print_function, division
import random, time, re, math, unicodedata, torch, numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s, e = 0, 1

class A:
    def __init__(self, x):
        self.x = x
        self.y = {}
        self.z = {}
        self.w = {0: "SOS", 1: "EOS"}
        self.n = 2

    def a(self, v):
        for u in v.split(' '):
            self.b(u)

    def b(self, v):
        if v not in self.y:
            self.y[v] = self.n
            self.z[v] = 1
            self.w[self.n] = v
            self.n += 1
        else:
            self.z[v] += 1

def u2a(v):
    return ''.join(c for c in unicodedata.normalize('NFD', v) if unicodedata.category(c) != 'Mn')

def nStr(v):
    v = u2a(v.lower().strip())
    v = re.sub(r"([.!?])", r" \1", v)
    v = re.sub(r"[^a-zA-Z!?]+", r" ", v)
    return v.strip()

def rL(x, y, z=False):
    print("Reading...")
    l = open('data/%s-%s.txt' % (x, y), encoding='utf-8').read().strip().split('\n')
    p = [[nStr(q) for q in o.split('\t')] for o in l]
    if z:
        p = [list(reversed(o)) for o in p]
        il = A(y)
        ol = A(x)
    else:
        il = A(x)
        ol = A(y)
    return il, ol, p

MX_L = 10
ep = ("i am ", "i m ", "he is", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re ")

def fP(p):
    return len(p[0].split(' ')) < MX_L and len(p[1].split(' ')) < MX_L and p[1].startswith(ep)

def fp(x):
    return [i for i in x if fP(i)]

def pD(x, y, z=False):
    il, ol, p = rL(x, y, z)
    print("Read %s pairs" % len(p))
    p = fp(p)
    print("Trimmed to %s pairs" % len(p))
    for pair in p:
        il.a(pair[0])
        ol.a(pair[1])
    print(il.x, il.n)
    print(ol.x, ol.n)
    return il, ol, p

il, ol, p = pD('eng', 'fra', True)

def iS(l, v):
    return [l.y[w] for w in v.split(' ')]

def tF(l, v):
    ids = iS(l, v)
    ids.append(e)
    return torch.tensor(ids, dtype=torch.long, device=d).view(1, -1)

def tFP(p):
    inT = tF(il, p[0])
    tT = tF(ol, p[1])
    return (inT, tT)

def gDL(b):
    il, ol, p = pD('eng', 'fra', True)
    n = len(p)
    iD = np.zeros((n, MX_L), dtype=np.int32)
    tD = np.zeros((n, MX_L), dtype=np.int32)
    for i, (inP, tP) in enumerate(p):
        inIds = iS(il, inP)
        tIds = iS(ol, tP)
        inIds.append(e)
        tIds.append(e)
        iD[i, :len(inIds)] = inIds
        tD[i, :len(tIds)] = tIds
    trD = TensorDataset(torch.LongTensor(iD).to(d), torch.LongTensor(tD).to(d))
    trS = RandomSampler(trD)
    trDL = DataLoader(trD, sampler=trS, batch_size=b)
    return il, ol, trDL

def tE(dL, enc, dec, encOpt, decOpt, crit):
    totL = 0
    for d in dL:
        inT, tT = d
        encOpt.zero_grad()
        decOpt.zero_grad()
        encOut, encHid = enc(inT)
        decOuts, _, _ = dec(encOut, encHid, tT)
        loss = crit(decOuts.view(-1, decOuts.size(-1)), tT.view(-1))
        loss.backward()
        encOpt.step()
        decOpt.step()
        totL += loss.item()
    return totL / len(dL)

def asMin(x):
    m = math.floor(x / 60)
    x -= m * 60
    return '%dm %ds' % (m, x)

def tSince(s, p):
    n = time.time()
    x = n - s
    eS = x / p
    rS = eS - x
    return '%s (- %s)' % (asMin(x), asMin(rS))

def showP(p):
    plt.figure()
    fig, ax = plt.subplots()
    loc = tkr.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(p)
    plt.show()

def train(dL, enc, dec, e, lr=0.001, pe=100, plo=100):
    st = time.time()
    plL = []
    plLTotal = 0
    prLTotal = 0
    encOpt = optim.Adam(enc.parameters(), lr=lr)
    decOpt = optim.Adam(dec.parameters(), lr=lr)
    crit = nn.NLLLoss()
    for i in range(1, e + 1):
        l = tE(dL, enc, dec, encOpt, decOpt, crit)
        prLTotal += l
        plLTotal += l
        if i % pe == 0:
            prLAvg = prLTotal / pe
            prLTotal = 0
            print('time: %s remain: (%d %d%%) Loss: %.4f' % (tSince(st, i / e), i, i / e * 100, prLAvg))
        if i % plo == 0:
            plAvg = plLTotal / plo
            plL.append(plAvg)
            plLTotal = 0
            showP(plL)

def evaluate(enc, dec, s, inL, outL):
    with torch.no_grad():
        inT = tF(inL, s)
        encOuts, encHid = enc(inT)
        decOuts, decHid, _ = dec(encOuts, encHid)
        _, topi = decOuts.topk(1)
        decIds = topi.squeeze()
        decW = []
        for i in decIds:
            if i.item() == e:
                decW.append('<EOS>')
                break
            decW.append(outL.w[i.item()])
    return decW, _

def evalRand(enc, dec, n=10):
    for _ in range(n):
        p = random.choice(p)
        print('>', p[0])
        print('=', p[1])
        outW, _ = evaluate(enc, dec, p[0], il, ol)
        outS = ' '.join(outW)
        print('<', outS)
        print('')

def showAttn(inS, outW, attn):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + inS.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + outW)
    ax.xaxis.set_major_locator(tkr.MultipleLocator(1))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(1))
    plt.show()

def evalShowAttn(inS, enc, dec):
    outW, attn = evaluate(enc, dec, inS, il, ol)
    print('input =', inS)
    print('output =', ' '.join(outW))
    showAttn(inS, outW, attn[0, :len(outW), :])

hS = 128
bS = 32
il, ol, tDL = gDL(bS)
enc = EncoderRNN(il.n, hS).to(d)
dec = AttnDecoderRNN(hS, ol.n).to(d)
train(tDL, enc, dec, 100, pe=5, plo=5)
enc.eval()
dec.eval()
evalRand(enc=enc, dec=dec, n=10)
evalShowAttn('il n est pas aussi grand que son pere', enc=enc, dec=dec)
evalShowAttn('je suis trop fatigue pour conduire', enc=enc, dec=dec)
evalShowAttn('je suis desole si c est une question idiote', enc=enc, dec=dec)
evalShowAttn('je suis reellement fiere de vous', enc=enc, dec=dec)
