import matplotlib.pyplot as plt

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append(row.split(','))
    return final_list
# path_train_label = 'F:/MYfile/Machine Learning/Homework/HW2/hw2_ml/PLA_b/train_label.csv'
path_train_data = 'train_data.csv'
path_train_label = 'train_label.csv'
train_data = read_csv(path_train_data)
train_label = read_csv(path_train_label)
N = len(train_data)-1
M = len(train_data[1])+1
train_data = train_data[1:N]
train_label = train_label[1:N]
N = N-1

X = []
Y = []
x_train_0 = []
y_train_0 = []
x_train_1 = []
y_train_1 = []
for i in range(N):
    Y.append(int(train_label[i][0]))
    X.append([])
    for j in range(M):
        if j == 0:
            X[i].append(1)
        else:
            X[i].append(float(train_data[i][j-1]))
    if Y[i] == 0:
        x_train_0.append(X[i][1])
        y_train_0.append(X[i][2])
    else:
        x_train_1.append(X[i][1])
        y_train_1.append(X[i][2])

x_train = []
y_train = []
for i in range(N):
    x_train.append(X[i][1])
    y_train.append(X[i][2])
max_x = max(x_train)
min_x = min(x_train)
x_line = list(range(int(min_x)-1, int(max_x)+1, 1))
plt.figure(1)
plt.title('PLA-a')
plt.xlabel('xlabel:x(1)')
plt.ylabel('ylabel:x(2)')
plt.scatter(x_train_0, y_train_0, label='y-label=0 in training set')
plt.scatter(x_train_1, y_train_1, label='y-label=1 in training set')

class Hw1List:

    def plus0(self, x, a):
        list_plus00 = list(map(lambda x: x+a, x))
        return list_plus00

    def mult0(self, x, y):
        list_sum = sum(list(map(lambda x, y: x*y, x, y)))
        return list_sum

    def time0(self, x, a):
        list_time0 = list(map(lambda x: a*x, x))
        return list_time0

    def plus1(self, x, y):
        list_plus0 = list(map(lambda x, y: x+y, x, y))
        return list_plus0

    def sign0(self, x):
        x = float(x)
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def NumNeq(self, x, y):
        NumNeq = sum(list(map(lambda x, y: x != y, x, y)))
        return NumNeq



Hw1 = Hw1List()
w = []
for i in range(M):
    w.append(float(0))
E_t = []
t = 0


def ELabel(w, X, Y):
    N = len(Y)
    ErrorLabel = []
    for i in range(N):
        h = 0.5*(Hw1.sign0(Hw1.mult0(X[i], w)) + 1)
        if h != Y[i]:
            ErrorLabel.append(i)
    return ErrorLabel


def WInteration(w, X, Y, w_recode):
    ErrorLabel = ELabel(w, X, Y)
    NumError = len(ErrorLabel)
    for i in range(NumError):
        w_t = Hw1.plus1(Hw1.time0(X[ErrorLabel[i]], (2*Y[ErrorLabel[i]] - 1)), w)
        NumError_t = len(ELabel(w_t, X, Y))
        if NumError_t < NumError:
            w_recode.append(w_t)
            WInteration(w_t, X, Y, w_recode)
            break
    return w_recode


w_recode = []
w_recode = WInteration(w, X, Y, w_recode)
NumLine = len(w_recode)
E_in = []
for i in range(NumLine):
    w = w_recode[i]
    E_in.append(len(ELabel(w, X, Y)))
    y = Hw1.plus0(Hw1.time0(x_line, -w[1]/w[2]), -w[0]/w[2])
    plt.plot(x_line, y, label='$h^t$ at t={}'.format(i+1))
    print('h^t=0.5*(sign({0:.2f}+({1:.2f}x(1))+({2:.2f}x(2)))+1) at t={3}'.format(w[0], w[1], w[2], i+1))


path_text_data = 'test_data.csv'
path_text_label = 'test_label.csv'
test_data = read_csv(path_text_data)
test_label = read_csv(path_text_label)
N_t = len(test_data)-1
M_t = len(test_data[1])+1
test_data = test_data[1:N_t]
test_label = test_label[1:N_t]
N_t = N_t-1

X_t = []
Y_t = []
x_test_0 = []
y_test_0 = []
x_test_1 = []
y_test_1 = []
for i in range(N_t):
    Y_t.append(int(test_label[i][0]))
    X_t.append([])
    for j in range(M_t):
        if j == 0:
            X_t[i].append(1)
        else:
            X_t[i].append(float(test_data[i][j-1]))
    if Y_t[i] == 0:
        x_test_0.append(X_t[i][1])
        y_test_0.append(X_t[i][2])
    else:
        x_test_1.append(X_t[i][1])
        y_test_1.append(X_t[i][2])

x_test = []
y_test = []
for i in range(N_t):
    x_test.append(X_t[i][1])
    y_test.append(X_t[i][2])

plt.scatter(x_test_0, y_test_0, c='r', marker='*', label='y-label=0 in test set')
plt.scatter(x_test_1, y_test_1, c='b', marker='*', label='y-label=1 in test set')
plt.legend(loc='lower right')

plt.figure(2)
plt.title('In-sample error versus iteration time')
plt.xlabel('iteration times')
plt.ylabel('in-sample error')
plt.plot(E_in, 'bo-')

plt.show()
