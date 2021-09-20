#!/usr/bin/env python3

"""
    File to write a hashed implementation of a Quadtree and display it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

    def __init__(self, index, val=None) -> None:
        self.index = index
        self.val = val

    def update(self, value) -> None:
        # TODO: this will have to update considering the value is "None" or something
        # it will be none if CHILDREN have been previously inserted
        # self.val -=.1
        nval = np.asarray(self.val) * np.asarray(value)
        self.val = tuple(nval)
        # print("Got to update {}. Currently holds {}, type: {}, incoming values are: {}, type: {}".format(self.index, self.val, type(self.val), value, type(value)))
    
    def insert(self, val) -> None:
        self.val = val

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


class Quadtree:

    MAX_DEPTH = 16
    BIT = 4
    OFFSET = 2

    def __init__(self, low = (0., 0.), scale = 1.0, max_depth = 4) -> None:
        self.dictionary = {1: QuadtreeElement(index=1)}
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
            Code to insert a value into the dictionary by index
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
        # if the index does not exist, create it with default None (if no value is given - this is mainly for mothers)
        if idx not in self.dictionary:
            self.dictionary[idx] = QuadtreeElement(index=idx, val=value)
        # if it exists, but it's None, but the new value isn't, put in the new value.
        if value is not None:
            self[idx].insert(value) if self[idx].val is None else self[idx].update(value)
        
    def insert_points(self, idx_val_dict: dict) -> None:
        for k, v in idx_val_dict.items():
            self.insert_point(k, tuple(v) if v is not None else v)

    def update_idx(self, idx, ch_num=None):
        """
            updating an index
        """
        it = self[idx]
        it.update(ch_num=ch_num)

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
            idcs_dict[midx] = None if f_sidcs else idcs_dict[idx] 
            del idcs_dict[idx]
            frontier.append(midx)
            if midx == 1:
                break
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



    def get_current_depth(self):
        return self.current_depth
    
    def ismaxdepth(self) -> bool:
        return True if self.current_depth >= self.max_levels else False
    
    def getBox(self, idx):
        """
            Implementation 1:1 from the book
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
            r = Rectangle(lo, w, h, facecolor=self[k].val if self[k].val is not None else 'none', edgecolor='red', lw=1, alpha=0.3)
            ax.add_patch(r)
    
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

    @classmethod
    def getMaxBoxes(cls,l=1):
        return ((4**l) - 1) / 3