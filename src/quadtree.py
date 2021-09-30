#!/usr/bin/env python3

"""
    File to write a hashed implementation of a Quadtree and display it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

from geometry_msgs.msg import Point32
import pickle

class Point:

    def __init__(self, x = 0., y = 0.) -> None:
        
        self.x = (x, y)
    
    def insideBox(self, box):
        """
            boolean test to decide whether the point is inside a given box
        """
        for i in range(len(box.lo)):
            if self.x[i] < box.lo[i] or self.x[i] > box.hi[i]: return False
        return True


class Box:

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
    
    def matplotlib_format(self):
        width = self.hi[0] - self.lo[0]
        height = self.hi[1] - self.lo[1]
        return self.lo, width, height

class QuadtreeElement:

    @classmethod
    def instantiate(cls, prior = [0.5, 0.5], sensor_model=[[0.7, 0.3], [0.3, 0.7]], clamp=6):
        """
            Method to instantiate class variables to be used for updating the tree. 
            Has to be called before inserting the first element
        """
        prior = np.asarray(prior)
        sensor_model = np.asarray(sensor_model)
        
        # log forms
        cls.sensor_model = np.log( sensor_model / (1. - sensor_model))
        cls.init_prior = np.log(( prior / (1. - prior)))

        # clamping values - are an index        
        cls.clamp = cls.initclamp(clamp)

    @classmethod
    def initclamp(cls, clamp):
        p = cls.init_prior
        for _ in range(clamp):
            model = cls.sensor_model
            obs = np.squeeze(model[:, 1])
            init_pr = cls.init_prior
            p = p + obs - init_pr
        return p

    def __init__(self, index, val=None) -> None:
        """
            Initialisation of an empty Node - empty, because val == None - used for checks
        """
        self.index = index
        self.val = val
        # self.val = val

    def update(self, value, pr = None) -> None:
        """
            Using the log update rule. Does not require a prior. If a prior is given, this will be used to update the value
            Used with the normal quadtree insertion
        """
        if pr is None:
            pr = self.val
        nval = self.updateVal(value, pr)
        self.val = nval

    @classmethod
    def updateVal(cls, val, pr):
        """
            Class method to just calculate the updated value
        """
        model = cls.sensor_model
        obs = np.squeeze(model[:, val])
        init_pr = cls.init_prior
        nval = pr + obs - init_pr
        nval = cls.clamping(nval)
        return nval

    @classmethod
    def clamping(cls, val):
        """
            Following the definition from the Octomaps paper
        """
        return np.clip(val, cls.clamp[0], cls.clamp[1])


    def insertNew(self, val) -> None:
        """
            val is either 0.0 or 1.0. The new value should be updated in a bayesian fashion
        """
        model = type(self).sensor_model
        obs = np.squeeze(model[:,val])
        self.val = obs

    def override(self, idx, val):
        """
            method to override both the value and the index
        """
        self.val = val
        self.index = idx 
    
    def overrideVal(self, val):
        """
            Method to override the value, not the index
        """
        self.val = val

    def overrideIdx(self, idx):
        """
            Method to override the index, not the value
        """
        self.index = idx


    def getprobabilities(self):
        """
            Function to turn the log-odds back into probabilities (belief)
        """
        return (1. - (1. / (1. + np.exp(self.val)))) # if self.val is not None else None

    def getMaxProbability(self):
        """
            Function to get the probability at its maximum index
            only works if val is not none
        """
        return self.getprobabilities()[self.getMaxVal()]


    def getlogprobs(self):
        """
            Function to return the log-odds
        """
        return self.val

    def getMaxVal(self):
        """
            Function to get the argmax - basically whether it is an interesting class or not. 
        """
        return np.argmax(self.val)


    def __repr__(self) -> str:
        return "{}, v: {}".format(self.index, self.val)
            
    def __str__(self) -> str:
        return self.__repr__()

    def getlevel(self):
        """
            function to calculate the depth of element. Mathematical equation. O(1). Coming from the geometric series that idx <= (1-4^n) / (1-4). Solve for n gives ceil(log4(idx * 3 + 1))
        """
        out = np.log(1 + 3 * self.index) / np.log(4)
        return np.ceil(out) 

    # make JSON serializable
    def __json__(self):
        return {
            self.index: self.val
        }

    for_json = __json__

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, json):
        obj = cls
        obj.index = json.index
        obj.val = json[obj.index]
        return obj


class Quadtree:

    MAX_DEPTH = 16
    BIT = 4
    OFFSET = 2

    def __init__(self, low = (0., 0.), scale = 1.0, max_depth = 4) -> None:
        QuadtreeElement.instantiate()
        self.dictionary = {1: QuadtreeElement(index=1)}
        # see if this works
        self[1].val = self[1].init_prior
        if max_depth > self.MAX_DEPTH:
            raise ValueError("{} too deep for Implementation - {} is maximum".format(self.max_depth, self.MAX_DEPTH))
        self.max_depth = max_depth
        self.current_depth = 1

        # Scaling the tree - done differently in the original implementation
        self.low = low
        self.scale = scale

    def __repr__(self) -> str:
        return "QuadTree. Max depth: {}, Scale: {}".format(self.max_depth, self.scale)

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self):
        return len(self.dictionary)

    def find_idx(self, point) -> int:
        """
            finds the smallest index for a point to be inserted into. Does not check whether this is occupied or not.
        """
        # descneding through the levels - Depth (only) tree search
        b = 1
        for p in range(2, self.max_depth+1):
            ds = self.getalldaughters(b)
            for d in ds:
                # This is the point where multiple points could be checked. 
                if point.insideBox(self.getBox(d)):
                    b = d
                    break
            if b > ds[-1]: break
        return b

    def find_idcs(self, pts_dict: dict) -> dict:
        idcs = {}
        for pt, val in pts_dict.items():
            if not isinstance(pt, Point):
                pt = Point(pt[0], pt[1])
            idcs[self.find_idx(pt)] = val
        return idcs
        
    def insert_point(self, idx, val: tuple):
        """
            Code to insert a value into the dictionary by index.
        """

        # Insert the child with the actual value
        self.insert_idx(idx, val)
        midx = self.getmother_idx(idx)
        while (midx >= 1):
            # traverse through the tree and insert the mothers with a NONE value
            self.insert_idx(midx)
            midx = self.getmother_idx(midx)
            

    def insert_idx(self, idx, value=None):
        """
            Basic function to insert a value into a specific index.
            Performs a check whether the index already exists
        """
        # if the index does not exist, create it. If no value is given, use default None (if no value is given - this is mainly for mothers)
        if idx not in self.dictionary:
            self.dictionary[idx] = QuadtreeElement(index=idx, val=value)
        # if it exists, but it's None, but the new value isn't, put in the new value.
        if value is not None:
            self[idx].insert(value) if self[idx].val is None else self[idx].update(value)
        
    def insert_points(self, idx_val_dict: dict) -> None:
        for k, v in idx_val_dict.items():
            self.insert_point(k, v)

    def insert_points_arr(self, values, idcs):
        """
            Function to insert the quadtree elements at the given position with a given prior
        """
        for i, val in enumerate(values):
            self.insert(idcs[i], val)

    def getPrior(self, idx):
        """
            Function to return a prior for a certain index
        """
        while (idx > 0):
            if idx in self.dictionary:
                prior = self[idx].val
                break
            idx = self.getmother_idx(idx)
        return prior

    def insert(self, idx, val):
        """
            Function to insert value. If idx does not exist, create new. If it exists, then use the value to update it
        """
        prior = self.getPrior(idx)
        if idx in self.dictionary:
            self.dictionary[idx].update(val, pr=prior)           
        else:
            nval = QuadtreeElement.updateVal(val, prior)
            self.dictionary[idx] = QuadtreeElement(idx, nval)

    def update_idx(self, idx, value=None):
        """
            updating an index
        """
        it = self[idx]
        it.update(value=value)

    def __getitem__(self, idx):
        """
            Function to return an element
            Does not do any checking
            Written this way to update to more complex data structures
        """
        return self.dictionary[idx]
    
    # ? currently unsued
    def split_idx(self, idx):
        """
            Function to split an index because there will be another observation at the lower level
        """
        idcs = self.getalldaughters(idx)
        ptval = self.dictionary[idx]
        for idx in idcs:
            pass

    
    def get_tree_idcs(self, idx):
        """
            Gets the indices of the tree from the current node to root
        """

        midx = idx
        idcs = []
        while (midx > 1):
            idcs.append(midx)
            midx=self.getmother_idx(midx)
        return idcs

    # pruning procedure - is a container
    def prune(self, leaf_idcs):
        """
            Pruning procedure for the tree. Requires the indices of the last inserted leaves
        """

        # ! this is like a breadth first tree search by Gavin - pop
        while list(leaf_idcs):
            lidx = leaf_idcs.pop(0)
            # midx = lidx
            s_idcs = self.getallsiblings(lidx)
            # if any sibling already exist in the dictionary
            if any(s_idx in self.dictionary for s_idx in s_idcs):
                # ! this is where the mother idcs should be updated iteratively, with the same condition - should hold a "subdivided" boolean - that allows them to be displayed
                pass
            else:
                ch_num = self.getchildinmother_numb(lidx)
                midx = self.getmother_idx(lidx)
                # TODO: Write this function that will update the mother 
                self.update_idx(midx, ch_num)
                try:
                    del self.dictionary[lidx]
                except KeyError:
                    print("Key: {} does not exist anymore".format(lidx))
                leaf_idcs.append(midx)

    # Reducing the indices - is similar to the pruning step, but operates before insertion
    def reduce_idcs(self, idcs_dict : dict):
        """
            Reducing the indices to check whether they can be added in at a higher level
        """
        reduced_idcs = {}
        frontier = list(idcs_dict.keys())
        while len(frontier) > 0:
            idx = frontier.pop(0)
            # ! you dumb idiot
            # TODO: simpler check get the mothers for all elements - where the mothers have duplicate entries, there we insert the children
            # ! come on.
            sidcs = self.getallsiblings(idx)
            f_sidcs = [sidx for sidx in sidcs if sidx in frontier]
            # if any siblings, the siblings should also be inserted
            if f_sidcs:
                reduced_idcs[idx] = idcs_dict[idx] if idx in idcs_dict else print("{} does not have a value in the tmp_dictionary".format(idx))
                # add the siblings to the dictionary too, if they exist
                for f_idx in f_sidcs:
                    reduced_idcs[f_idx] = idcs_dict[f_idx] if f_idx in idcs_dict else print("Sibling {} does not have a value".format(f_idx))
                    frontier.remove(f_idx)
            # ALWAYS add the mother to the search frontier - so that other inserted nodes will know if there is something at a deeper depth
            midx = self.getmother_idx(idx)
            if midx <= 1:
                break
            idcs_dict[midx] = None if f_sidcs else idcs_dict[idx] 
            del idcs_dict[idx]
            frontier.append(midx)
        return reduced_idcs

    def reduce_idcs_alternative(self, idcs_dict: dict) -> dict:
        """
            With the help of this:  https://stackoverflow.com/questions/25264798/checking-for-and-indexing-non-unique-duplicate-values-in-a-numpy-array
        """
        reduced_idcs = {}
        frontier = list(idcs_dict.keys())
        while len(frontier) > 0:        #! change this condition
            mothers = np.asarray([self.getmother_idx(idx) for idx in frontier])
            unq, unq_idx, unq_cnt = np.unique(mothers, return_inverse=True, return_counts=True)
            cnt_mask = unq_cnt > 1
            cnt_idx, = np.nonzero(cnt_mask)
            idx_mask = np.in1d(unq_idx, cnt_idx)
            idx_idx, = np.nonzero(idx_mask)
            srt_idx = np.argsort(unq_idx[idx_mask])
            dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
            print("test debug line")
            # TODO: continue here to think about whether this is a good idea / good approach.

    def insertion_idcs(self, pts : list, width : int, height : int):
        """
            Function that finds the indices where to insert. Already does the pruning.
            Requires a list of "Point" objects
        """
        assert len(pts) == (height * width)
        # initialise a big array where to insert the points
        arr = np.ones((len(pts),self.max_depth), dtype=np.int)

        # initialise a boolean list, that says whether to continue
        ct_vec = np.ones(len(pts),dtype=np.bool)

        # for every level
        for i in range(1, self.max_depth):
            # early stopping
            if not np.any(ct_vec):
                # fill in last column
                # arr[:,-1] = arr[:,i-1]
                # fill in all columns to the last column
                arr[:,i:] = arr[:,i-1,np.newaxis]
                break

            # actual work
            for j, pt in enumerate(pts):
                curr_box = arr[j,i-1].astype(np.int)

                # if the boolean vector is still valid
                if ct_vec[j]:
                    ds = self.getalldaughters(curr_box)
                    for d in ds:
                        pt_t = self.getPoint(pt)
                        if pt_t.insideBox(self.getBox(d)):
                            arr[j,i] = d
                            break
                # if the boolean vector says we should not continue
                else:
                    # just project the value forward
                    arr[j,i] = curr_box

            # now look at the neighborhood of each point
            for j, pt in enumerate(pts):
                own_val = arr[j,i]
                neighborhood_idcs = self.getNeighborhood(j, width=width, height=height) 
                # neighborhood_idcs = list(range(len(pts)))
                # neighborhood_idcs.pop(j)    # all but the own
                # ? could use the indices here in numpy form? Smart indexing? 
                ct = False
                for neighb_idx in neighborhood_idcs:
                    # we are still in level i
                    # if any of the neighbors has the same index - continue
                    # if none of the neighbors has the same index - stop
                    neighb_val = arr[neighb_idx, i].astype(np.int)
                    if neighb_val == own_val:
                        ct = True
                        break
                ct_vec[j] = ct

        # return the last row where the insertion indices **should** be held!
        return arr[:,-1]
        # return arr

    def getNeighborhood(self, idx : int, width : int, height : int):
        """
            Function to get the moore neighborhood with 1 of a pixel -> returns the 1D index
        """
        idcs = []
        r = idx // width
        c = idx % width
        for i in range(-1, 2):
            for j in range(-1, 2):
                idx = (r + i) * width + (c + j)
                if ((r+i) >= 0 and (r+i) < height and (c+j) >= 0 and (c+j) < width and ((i | j) != 0)):
                    idcs.append(idx)
        return idcs

    def get_current_depth(self):
        return self.current_depth
    
    def ismaxdepth(self) -> bool:
        return True if self.current_depth >= self.max_levels else False
    
    def getPoint(self, pt):
        """
            Function for correct typecasting
        """
        if isinstance(pt, tuple):
            return Point(pt[0], pt[1])
        elif isinstance(pt, Point32):
            return Point(pt.x, pt.y)

    def getBox(self, idx):
        """
            Implementation from the book
        """

        offset = [0.0, 0.0]
        delta = 1.0
        # Calculating the bounding box offsets
        while idx > 1:
            idx_b = (idx + self.OFFSET) % self.BIT
            for j in range(0,2):
                if (idx_b & (1 << j)): offset[j] += delta  
            idx = self.getmother_idx(idx)
            delta *= 2
        
        # Rescaling the offsets by delta - for correctness - 
        low = [0., 0.]
        high = [0., 0.]
        for j in range(0,2):
            low[j] = self.low[j] + self.scale * offset[j] / delta
            high[j] = self.low[j] + self.scale * (offset[j] + 1.) / delta
        
        return Box(low, high)

    def getmother_idx(self, idx):
        return (idx + self.OFFSET) >> 2

    def getchild_idx(self, idx, numb):
        """
            Function to get the index of a child given a number and an own index
        """
        return self.getfirstdaughter_idx(idx) + numb -1

    def getchildinmother_numb(self, idx):
        """
            Function to get the count of which offset.
            Will not work for idx 0
        """
        return (idx + self.BIT) % self.OFFSET

    def getfirstdaughter_idx(self, idx):
        # return self.BIT * idx - self.OFFSET    # Alternative
        return 4 * idx - 2
    
    def getlastdaughter_idx(self, idx):
        # return self.getfirstdaughter_idx() + self.BIT - 1 # Alternative
        return 4 * idx + 1
    
    def getseconddaughter_idx(self, idx):
        return self.getfirstdaughter_idx(idx) + 1
    
    def getthirddaughter_idx(self, idx):
        return self.getfirstdaughter_idx(idx) + 2
    
    def getalldaughters(self, idx):
        daughters = [i for i in range(self.getfirstdaughter_idx(idx), self.getlastdaughter_idx(idx) + 1)]
        return daughters
    
    def getallsiblings(self, idx):
        midx = self.getmother_idx(idx)
        sibl = self.getalldaughters(midx)
        sibl.remove(idx)
        return sibl

    def getallboxes(self):
        bdict = {}
        for k in self.dictionary.keys():
            bdict[k] = self.getBox(k)
        return bdict

    def plot_tree(self, ax):
        for k, v in self.getallboxes().items():
            lo, w, h = v.matplotlib_format()
            # alpha = 0 if self[k].val is None else self[k].getMaxProbability()
            r = Rectangle(lo, w, h, facecolor='blue' if self[k].getMaxVal() == 1 else 'none', edgecolor='black', lw=.2 ) # , alpha=alpha)
            ax.add_patch(r)
        ax.set_xlim(self.low[0], self.low[0] + self.scale)
        ax.set_ylim(self.low[1], self.low[1] + self.scale)
    
    def getMotherChain(self, idx):
        idcs = {}
        while(idx >1):
            midx = self.getmother_idx(idx)
            idcs[idx] = midx
            idx = midx
        return idcs

    def printMotherChain(self, idcs):
        for idx in idcs:
            [print("{}: {}".format(k,v)) for k,v in self.getMotherChain(idx).items()]

    def printvals(self):
        [print("{}: {}".format(k, v.val)) for k, v in self.dictionary.items()]

    def getlevel(self, idx):
        """
            function to calculate the depth of an element. Mathematical equation. O(1). Coming from the geometric series that idx <= (1-4^n) / (1-4). Solve for n gives ceil(log4(idx * 3 + 1))
        """
        out = np.log(1+ 3 * idx) / np.log(4)
        return np.ceil(out)

    def getlevel_log(self, idx):
        """
            logarithmic calculation of the depth of an element. Tree search. O(log(n))
        """
        nds = 0
        for i in range(1,self.max_depth+1):
            nds += self.BIT ** (i-1)
            if nds >= idx:
                return i    
        raise ValueError("maxdepth reached. node is deeper than possible levels")

    def find_priors(self, insertion_idcs_dict : dict) -> dict:
        """
            Method to find the priors by recursively going through the mothers.
            Return the lowest level prior that can be found
        """
        return_dict = {}
        for k in insertion_idcs_dict.keys():
            midx = k
            while(midx > 1):
                midx = self.getmother_idx(midx)
                if midx in self.dictionary and self[midx].val is not None:
                    return_dict[k] = self[midx].getlogprobs()
        return return_dict

    def find_priors_arr(self, insert_arr, prior_size=2):
        """
            Method to find the priors by recursively walking back up through the vector
        """
        ret_arr = np.ones((insert_arr.size, prior_size))
        for i in range(insert_arr.size):
            midx = insert_arr[i]
            while (midx > 0):
                if midx in self.dictionary:
                    ret_arr[i] = self[midx].val
                    break
                midx = self.getmother_idx(midx)
        return ret_arr


    def save(self, fpath):
        """
            Function to save the quadtree as a pickle file
        """
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)
        return "saved file to {}".format(fpath)

    @classmethod
    def load(cls, fpath):
        with open(fpath, 'rb') as f:
            tree = pickle.load(f)
        return tree
    
    @classmethod
    def getMaxBoxes(cls,l=1):
        return ((4**l) - 1) / 3

    @classmethod
    def getIndicesPerLevel(cls, l):
        """
            Function to return the indices for each level
        """
        idcs = cls.getMaxBoxes(l)
        b_idcs = cls.getMaxBoxes(l-1)
        return (int(b_idcs), int(idcs))

    def postprocess(self):
        """
            Postprocessing function to prune the tree depending on inserted values
            May take really long.
        """
        # recurse through the levels
        for l in range(self.max_depth, 2, -1):
            # Get the indices for that level
            idcs = self.getIndicesPerLevel(l)
            # Run through all of the keys
            orig_keys = list(self.dictionary.keys())
            for k in orig_keys:
                if (k > idcs[0]) and (k < idcs[1]):
                    if k in self.dictionary:
                        # Look at all the siblings that exist in the dictionary
                        siblings = [sib for sib in self.getallsiblings(k) if sib in self.dictionary]
                        # Get the probability
                        own_val = self[k].getlogprobs()
                        # sib_prob = []
                        # if siblings:
                        #     for sib in siblings:
                        #         lp = self[sib].getlogprobs()
                        #         sib_prob.append(lp)
                        sib_prob = [self[sib].getlogprobs() for sib in siblings if siblings and self[sib].getlogprobs() is not None]
                        if own_val is not None:
                            sib_prob.insert(0, own_val)

                        # only run the test if the list is not empty
                        if sib_prob:
                            # Test with the postprocess_equality function - if all children are reasonably equal
                            if self.postprocess_equality(sib_prob):
                                # replace the mother with the average of all the children
                                midx = self.getmother_idx(k)
                                m_new = np.asarray(sib_prob).mean(axis=0)
                                # if the mother already exists, check if it is bigger than the mean
                                if midx in self.dictionary:
                                    m_val = self[midx].getlogprobs()
                                    if abs(m_val[0]) < abs(m_new[0]):
                                        self.postprocess_insert(midx, m_new)
                                else:
                                    self.postprocess_insert(midx, m_new)


                                # for sib in siblings del self.dictionary[sib]
                            # del [self.dictionary[sib] for sib in siblings]
                                self.postprocess_clean(midx)        
                        # else only gets hit if all the children are "none" values
                        else:
                            self.postprocess_clean(midx)


    def postprocess_clean(self, idx):
        # clean up all the daughters
        chds = self.getalldaughters(idx)
        for ch in chds:
            if ch in self.dictionary:
                del self.dictionary[ch]

    def postprocess_insert(self, idx, val):
        """
            function to make the postprocess - insert step easier.
            inserts the value at idx and deletes all children
        """
        if idx in self.dictionary:
            self.dictionary[idx].overrideVal(val)
        else:
            self.dictionary[idx] = QuadtreeElement(idx, val)
        

                    
    def postprocess_equality(self, sib_vals):
        """
            Function to test the postprocessing equality. Has to return a bool.
            If we are working with the log-odds, the sign is the same for all of them
        """

        arr = np.asarray(sib_vals)
        s = np.sign(arr).sum(axis=0)
        if s[0] == arr.shape[0]:
            return True
        else:
            return False

    


        
