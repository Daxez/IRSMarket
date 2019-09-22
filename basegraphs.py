"""
Base graph classes: Graph, Node and edge. Written to be overriden

One of the reasons that the first attempt to simulate was so terribly slow
you could just plot it though which was pretty cool.
"""
# Author: Dexter Drupsteen
from __future__ import division

import random
import matplotlib.pyplot as pplot
import matplotlib.colors as mcolors
import networkx as nx

from uuid import uuid4

class Graph(nx.MultiDiGraph):

    def __init__(self, data=None, **attr):
        super(Graph,self).__init__(data,**attr)

    def add_edge(self,u,v,e):
        if(u == None or v == None or e == None):
            raise Exception("u,v nor e cannot be None.")

        # Just override val if exists, no harm in that, except for if you use
        # multiple graphs with the same node, you shouldn't do that! Added this
        # to enable 'get_edges' at node level.
        u.graph = self
        v.graph = self
        e.graph = self

        # Just so we have the info at hand
        e.left = u
        e.right = v

        # Let MultiDiGraph handle the rest
        super(Graph,self).add_edge(u,v,key=e.name,edge=e)

    def add_node(self,node):
        node.graph = self
        super(Graph,self).add_node(node)

    def average_degree(self):
        edges = len(self.edges())
        if(edges == 0):
            return 0

        return len(self.nodes())/edges

    def get_edges(self):
        # Yield for memory
        for e in self.edges(data=True):
            yield e[2]['edge']


    def draw(self,size=None,color=None,with_labels=False,filename=None):
        node_size = 300
        node_color = 'r'
        cdict = {'red':   ((0.0, 0.0, 0.0),
                           (0.50, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue':  ((0.0, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 1.0),
                           (0.33, 1.0, 1.0),
                           (1.0, 0.0, 0.0))}
        cmap = None

        if(size != None):
            if(size == "degree"):
                degree_dict = self.degree()
                degrees = [degree_dict[x] for x in degree_dict]
            else:
                degrees = [getattr(n,size) for n in self.nodes()]

            if(len(degrees) > 0):
                print degrees
                minval = min(min(degrees),0)
                ds = 600 / (max(degrees)-minval)
                node_size = [100 + int(ds*(x-minval)) for x in degrees]

        if(color != None):
            if(color == "degree"):
                degree_dict = self.degree()
                degrees = [degree_dict[x] for x in degree_dict]
            elif(color == "test"):
                degrees = xrange(len(self.nodes()))
            else:
                degrees = [getattr(n,color) for n in self.nodes()]

            if(len(degrees) > 0):
                minval = min(min(degrees),0)
                ds = 1.0 / (max(degrees)-minval)
                node_color = [ds*(x-minval) for x in degrees]
                #cmap = mcolors.LinearSegmentedColormap('RedToGreen',cdict, 100)
                cmap = pplot.get_cmap("YlOrRd")


        nx.draw_networkx(self,with_labels=with_labels,node_size=node_size,
                node_color=node_color,cmap=cmap,vmin=0,vmax=1)
        if(filename == None):
            pplot.show()
        else:
            pplot.savefig(filename)


class Edge(object):
    """
    Edge wrapper
    """

    def __init__(self, name=None):
        if(name == None):
            name = str(uuid4())
        self.name = name
        self.left = None
        self.right = None
        self.graph = None

    def remove(self):
        if(self.graph == None or self.left == None or self.right == None):
            raise Exception("Please add edge to a graph first (graph.add_edge)")

        self.graph.remove_edge(self.left, self.right,key=self.name)

class Node(object):
    """
    Node wrapper
    """

    def __init__(self, name=None):
        if(name == None):
            name = str(uuid4())
        self.name = name
        self.graph = None

    def get_edges(self):
        """
        Gets in and out edges for this node
        """
        if(self.graph == None):
            raise Exception("Please add node to a graph first")

        out = [x[2]['edge']
                for x in self.graph.out_edges(nbunch=[self],data=True)]
        _in = [x[2]['edge']
                for x in self.graph.in_edges(nbunch=[self],data=True)]

        return out,_in

    def get_out(self):
        if(self.graph == None):
            raise Exception("Please add node to a graph first")
        for x in self.graph.out_edges(nbunch=[self],data=True):
            yield x[2]['edge']

    def get_in(self):
        if(self.graph == None):
            raise Exception("Please add node to a graph first")
        for x in self.graph.in_edges(nbunch=[self],data=True):
            yield x[2]['edge']

    def degree(self):
        return list(self.graph.degree_iter(nbunch=[self]))[0][1]

    def degree_in(self):
        return list(self.graph.in_degree_iter(nbunch=[self]))[0][1]

    def degree_out(self):
        return list(self.graph.out_degree_iter(nbunch=[self]))[0][1]

    def remove(self):
        self.graph.remove_node(self)

if(__name__ == '__main__'):
    print "Hello world"
    print "Going to generate random network for plotting purposes now."
    graph = Graph()

    nodes = []
    n = 100
    node_a = Node()
    node_b = Node()

    graph.add_edge(node_a,node_b,Edge())
    print graph.edges(data=True)
    print graph.get_edges()

    graph.draw()
