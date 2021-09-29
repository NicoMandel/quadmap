#!/usr/bin/env python3

"""
    Standard implementation of a quadtree
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

from geometry_msgs.msg import Point32
import pickle

from quadtree import QuadtreeElement, Point, Box

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

    # Magic Methods
    def __repr__(self) -> str:
        return "QuadTree. Max depth: {}, Scale: {}".format(self.max_depth, self.scale)

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, idx):
        """
            Function to return an element
            Does not do any checking
            Written this way to update to more complex data structures
        """
        return self.dictionary[idx]

    def __len__(self):
        return len(self.dictionary)
    
    # Saving and loading
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
    
    # Level calculations
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

    def getlevel(self, idx):
        """
            function to calculate the depth of an element. Mathematical equation. O(1). Coming from the geometric series that idx <= (1-4^n) / (1-4). Solve for n gives ceil(log4(idx * 3 + 1))
        """
        out = np.log(1+ 3 * idx) / np.log(4)
        return np.ceil(out)

    # Getting Indices 
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

    # Getting Objects
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
     
    def getPoint(self, pt):
        """
            Function for correct typecasting
        """
        if isinstance(pt, tuple):
            return Point(pt[0], pt[1])
        elif isinstance(pt, Point32):
            return Point(pt.x, pt.y)

    # Entire Tree Operations
    def getallboxes(self):
        bdict = {}
        for k in self.dictionary.keys():
            bdict[k] = self.getBox(k)
        return bdict

    def plot_tree(self, ax):
        for k, v in self.getallboxes().items():
            lo, w, h = v.matplotlib_format()
            # alpha = 0 if self[k].val is None else self[k].getMaxProbability()
            r = Rectangle(lo, w, h, facecolor='blue' if self[k].getMaxVal() == 1 else 'none', edgecolor='red', lw=.5 ) # , alpha=alpha)
            ax.add_patch(r)
        ax.set_xlim(self.low[0], self.low[0] + self.scale)
        ax.set_ylim(self.low[1], self.low[1] + self.scale)

    def printvals(self):
        [print("{}: {}".format(k, v.val)) for k, v in self.dictionary.items()]

    # Tree traversal
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

    # Depth checkers
    # ? unused? 
    def get_current_depth(self):
        return self.current_depth
    
    def ismaxdepth(self) -> bool:
        return True if self.current_depth >= self.max_levels else False
    
    # Finding index - generic
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
    
    # ! **2** - inserting the value list into the indices - for a list
    def insert_points_arr(self, values, idcs):
        """
            Function to insert the quadtree elements at the given position with a given prior
        """
        for i, val in enumerate(values):
            self.insert(idcs[i], val)

    # ! **2.1** - inserting a single value into the index. Not for a list, but for a single object.
    def insert(self, idx, val):
        """
            Function to insert value. If idx does not exist, create new. If it exists, then use the value to update it
        """
        if idx in self.dictionary:
            self.dictionary[idx].update(val)           
        else:
            # Creates an object with value none
            self.dictionary[idx] = QuadtreeElement(idx)
            # Set the value according to the sensor model
            self.dictionary[idx].insertNew(val)

    # ! **1** - entry point for the outside code
    def insertion_idcs(self, pts : list):
        """
            Function that finds the indices where to insert
            Goes to the deepest level
            Requires a list of "Point" objects
        """
        # initialise a big array where to insert the points
        arr = np.ones(len(pts), dtype=np.int)

        for i, pt in enumerate(pts):
            pt_t = self.getPoint(pt)
            idx = self.find_idx(pt_t)
            arr[i] = idx
        
        return arr
  
    # ! **2.2** - getting a prior
    def getPrior(self, idx):
        """
            Function to return a prior for a specific index
            Does not traverse the tree.
            Will fail if the index does not exist in the dictionary
        """
        prior = self[idx].getlogprobs()
        return prior


    # ! Pruning does not make sense - because everything will be inserted at leaf level - do pruning during post
    # Postprocessing
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
                        # TODO: figure out how to deal with the None values that come through the pickling
                        # ! may be fixed due to insertion
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
                                self.postprocess_insert(midx, m_new)
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
            inserts the value at idx
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
