'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, meta_info=False):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        self.meta_info = meta_info
        if meta_info:
            self.userInfo, self.trainInteractionLevel = self.load_user_info_file_as_matrix(path + ".user.info")
            self.itemInfo = self.load_item_info_file_as_matrix(path + ".item.info")
        else:
            self.userInfo, self.itemInfo, self.trainInteractionLevel = None, None, None
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat

    def load_user_info_file_as_matrix(self, filename):
        mat = np.zeros((self.num_users, 7))
        interactionLevel = list()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                uid = int(arr[0])

                mat[uid, int(arr[1])] = 1.0        # 0 ~ 5
                mat[uid, 6] = float(arr[2])        # 6
                interactionLevel.append(int(arr[3]))

                line = f.readline()
        return mat, interactionLevel
        
    def load_item_info_file_as_matrix(self, filename):
        mat = np.zeros((self.num_items, 72))
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                iid = int(arr[0])

                mat[iid, int(arr[1])] = 1.0             # 0 ~ 8
                mat[iid, int(arr[2]) + 9] = 1.0         # 9 ~ 15
                mat[iid, int(arr[3]) + 16] = 1.0        # 16 ~ 25
                mat[iid, int(arr[4]) + 26] = 1.0        # 26 ~ 42
                mat[iid, int(arr[5]) + 43] = 1.0        # 43 ~ 71
                line = f.readline()
        return mat