from math import sqrt, pow
from numpy import genfromtxt
import operator
import pandas as pd

def loadData():
    df = pd.read_csv("ratings.csv", error_bad_lines=False)
    return df

def loadMovies():
    df = pd.read_csv("movies.csv", error_bad_lines=False, header=1)
    return df

def user_sim_cosine_sim(person1, person2):
    multsum = 0
    sumA = 0
    sumB = 0
    for idx, val in enumerate(person1):
        if val != 0 and person2[idx] != 0:
            multsum = multsum+val*person2[idx]
            sumA = sumA + pow(val,2)
            sumB = sumB + pow(person2[idx],2)
    res = multsum/(sqrt(sumA) * sqrt(sumB))
    return res

def user_sim_pearson_corr(person1, person2):
    sum1 = 0
    sum2 = 0
    cnt = 0
    for idx,val in enumerate(person1):
        if val != 0 and person2[idx] != 0:
            cnt = cnt + 1
            sum1 = sum1 + val
            sum2 = sum2 + person2[idx]
    if sum1 != 0 or sum2 != 0:
        mean1 = sum1/cnt
        mean2 = sum2/cnt
        topsum = 0
        bottom1 = 0
        bottom2 = 0
        for idx, val in enumerate(person1):
            if val != 0 and person2[idx] != 0:
                topsum = topsum + (val - mean1)*(person2[idx] - mean2)
                bottom1 = bottom1 + pow((val - mean1),2)
                bottom2 = bottom2 + pow((person2[idx] - mean2), 2)

        if sqrt(bottom1)*sqrt(bottom2) != 0:
            res = topsum/(sqrt(bottom1)*sqrt(bottom2))
        else:
            res = -1
    else:
        res = -1
    return res

# computes similarity between two users based on the pearson similarity metric

def most_similar_users(vec1, vec2, dbase1, dbase2, weight1, number_of_users):
    weight2 = 1 - weight1
    similarities1 = {}
    similarities2 = {}
    for user in dbase1:
        similarities1[user[0]] = user_sim_pearson_corr(vec1[1:],user[1:])
    for user in dbase2:
        similarities2[user[0]] = user_sim_pearson_corr(vec2[1:],user[1:])

    max1 = max(similarities1.items(), key=operator.itemgetter(1))[1]
    max2 = max(similarities2.items(), key=operator.itemgetter(1))[1]

    for key,val in similarities1.items():
        similarities1[key] = similarities1[key]/max1
    for key,val in similarities2.items():
        similarities2[key] = similarities2[key]/max2
    for key, val in similarities1.items():
        similarities1[key] = similarities1[key]*weight1 + similarities2[key]*weight2
    newA = dict(sorted(similarities1.items(), key=operator.itemgetter(1), reverse=True)[:number_of_users+1])
    del newA[vec1[0]]
    return newA

# returns top-K similar users for the given

def user_recommendations(vec1, vec2, dbase1, dbase2, weight1):
    recs = []
    dset1 = {}
    for user in dbase1:
        dset1[user[0]] = user[1:]
    closest = most_similar_users(vec1, vec2, dbase1, dbase2, weight1, 10)
    for idx, val in enumerate(vec1[1:]):
        cnt = 0
        sum = 0
        if val == 0:
            for key, value in closest.items():
               if dset1[key][idx] != 0:
                sum = sum + dset1[key][idx]
                cnt = cnt + 1
            if cnt > 0:
                recs.append([idx, sum/cnt])
    print(sorted(recs, key = lambda x: x[1], reverse=True))
# generate recommendations for the given user

data = loadData()
dataset = []

for index, row in data.iterrows():
    row = [row[col] for col in list(data)]
    dataset.append(row)

movies = []
users = []
for row in dataset:
    movies.append(row[1])
    users.append(row[0])
movies = sorted(list(set(movies)))
users = sorted(list(set(users)))

nmovies = int(max(movies))

dbase = [[1] * (nmovies + 1)]

for user in users:
    dbase.append([0] * (nmovies + 1))
for idx,val in enumerate(users):
    dbase[idx+1][0] = val

for row in dataset:
    dbase[int(row[0])][int(row[1])] = row[2]

genres = {}

data = loadMovies()
movies = []
for index, row in data.iterrows():
    row = [row[col] for col in list(data)]
    movies.append(row)

for movie in movies:
    for genre in movie[2].split("|"):
        genres[genre] = ''

cnt = 1
for key,val in genres.items():
    genres[key] = cnt
    cnt += 1



ugenres = [[1]*cnt]
for user in dbase:
    ugenres.append([0]*cnt)

for idx,val in enumerate(dbase[1:]):
    print(idx)
    ugenres[idx+1][0] = val[0]
    for idx1,val1 in enumerate(val[1:]):
        if val1 != 0:
            for movie in movies:
                if movie[0] == idx1:
                    for genre in movie[2].split("|"):
                        ugenres[idx+1][genres[genre]] += 1

user_recommendations(dbase[2], ugenres[2], dbase, ugenres, 0)